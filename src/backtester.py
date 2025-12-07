import pandas as pd
import sys
import os
import traceback
from tqdm import tqdm
from datetime import timedelta

# 상위 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from config.settings import Config
from src.logger import setup_logger  # [NEW] 로거 사용

class Backtester:
    """
    과거 데이터 기반으로 시뮬레이션을 수행하는 타임머신 클래스
    (Logger 적용 및 Time Cut 로직 포함)
    """
    def __init__(self, df: pd.DataFrame, strategy, risk_manager):
        self.full_df = df
        self.strategy = strategy
        self.risk_manager = risk_manager
        
        # [NEW] 로거 초기화 (화면 출력 및 파일 저장 담당)
        self.logger = setup_logger()
        
        # 백테스트 상태 변수
        self.balance = Config.INITIAL_BALANCE
        self.position = None 
        self.trade_log = []     # 체결된 거래 기록
        self.equity_curve = []  # 자산 추이
        self.decision_log = []  # 전략의 모든 판단 기록
        
        # 수수료 설정
        self.fee_rate = Config.TRADING_FEE_RATE
        self.slippage = Config.SLIPPAGE

    def run(self):
        """백테스팅 메인 루프 실행"""
        
        # 1. 지표 계산 (전략 클래스에게 위임)
        self.full_df = self.strategy.calculate_indicators(self.full_df)

        # 2. 데이터 개수 검증 (타임프레임에 따라 하루 데이터 개수 추정)
        if 'h' in Config.TIMEFRAME:
            bars_per_day = 24 // int(Config.TIMEFRAME.replace('h',''))
        elif 'd' in Config.TIMEFRAME:
            bars_per_day = 1
        elif 'm' in Config.TIMEFRAME:
            bars_per_day = 1440 // int(Config.TIMEFRAME.replace('m',''))
        else:
            bars_per_day = 1 # 기본값

        # 테스트 기간에 필요한 최소 데이터 개수
        min_data_needed = Config.TEST_DAYS * bars_per_day
        
        # [수정됨] 조건문 완화 ( <= 를 < 로 변경)
        # 데이터가 정확히 180개여도 시뮬레이션은 가능해야 함
        if len(self.full_df) < min_data_needed:
            # 로그 레벨이 WARNING일 때만 출력되므로, 최적화 중에는 보이지 않을 수 있음.
            # 하지만 최적화 결과가 0이면 이 문제가 확실함.
            # 여기서는 Optuna 실행 시 너무 시끄럽지 않게 놔두거나 디버깅용 print 추가
            # print(f"DEBUG: Data length {len(self.full_df)} < Needed {min_data_needed}") 
            return None # [수정] 아무것도 반환하지 않음

        # 시작 인덱스 설정
        start_index = len(self.full_df) - min_data_needed
        start_index = max(0, start_index)
        
        self.logger.info(f"\n[Backtester] Starting simulation over last {Config.TEST_DAYS} days ({len(self.full_df)-start_index} bars)...")
        self.logger.info(f"Initial Balance: ${self.balance:,.2f}")
        
        # 3. 타임머신 루프 (Tqdm: 진행바 표시, Warning 레벨일 땐 숨김)
        show_progress = False if Config.LOG_LEVEL == "WARNING" else True
        
        for i in tqdm(range(start_index, len(self.full_df)), disable=not show_progress):
            
            # 과거 시점의 데이터 슬라이싱
            current_slice = self.full_df.iloc[:i+1]
            current_bar = current_slice.iloc[-1]
            
            # A. 포지션 관리 (청산 확인)
            if self.position:
                self._check_exit(current_bar)
            
            # B. 신규 진입 판단 (포지션 없을 때만)
            if not self.position:
                self._process_entry(current_slice)
            
            # C. 자산 기록
            self._record_equity(current_bar)

        # 4. 결과 리포트 생성
        report = self._generate_report()
        return report # [핵심 수정]

    def _process_entry(self, df_slice):
        """
        전략에게 시장 상황을 물어보고 진입 여부 결정
        """
        current_bar = df_slice.iloc[-1]
        timestamp = current_bar.name
        current_price = current_bar['close']
        
        log_entry = {
            "timestamp": timestamp,
            "price": current_price,
            "action": "HOLD",
            "reason": "No signal"
        }

        try:
            # 전략에게 신호 요청
            signal = self.strategy.generate_signal(df_slice)
            
            log_entry.update({
                "action": signal.action,
                "reason": signal.reason
            })
            
            # 매수/매도 신호 발생 시
            if signal.action in ["BUY", "SELL"]:
                # 리스크 매니저에게 수량 계산 요청
                params = self.risk_manager.calculate_entry_params(
                    balance=self.balance,
                    entry_price=signal.entry_price,
                    stop_loss=signal.stop_loss,
                    action=signal.action
                )
                
                if params:
                    params['trailing_stop_multiplier'] = signal.trailing_stop_multiplier                    
                    self._execute_entry(params, timestamp)
                    log_entry['status'] = "EXECUTED"
                else:
                    log_entry['status'] = "SKIPPED_RISK_CALC_FAIL"
            else:
                log_entry['status'] = "HOLD_BY_STRATEGY"
        
        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"⚠️ Error at {timestamp}: {error_msg}")
            self.logger.debug(traceback.format_exc())
            log_entry['reason'] = f"SYSTEM_ERROR: {error_msg}"
            
        finally:
            self.decision_log.append(log_entry)

    def _execute_entry(self, params, timestamp):
        """진입 주문 체결 시뮬레이션 (청산 추적 변수 추가)"""
        cost_rate = self.fee_rate + self.slippage
        entry_value = params['entry_price'] * params['quantity']
        fee = entry_value * cost_rate
        
        self.balance -= fee
        
        # 트레일링 스탑 정보 가져오기 (없으면 None)
        trailing_mult = params.get('trailing_stop_multiplier', None)

        self.position = {
            "type": params['action'],
            "entry_price": params['entry_price'],
            "quantity": params['quantity'],
            "stop_loss": params['stop_loss'],
            "take_profit": None, 
            "entry_time": timestamp,
            "trailing_mult": trailing_mult,
            
            # [NEW] 청산 관련 정보
            "liq_price": params['liq_price'], 
            "min_liq_dist_pct": 100.0 # 청산가까지 남은 최소 거리 (%) 초기화
        }
        
        action_icon = "🟢" if params['action'] == "BUY" else "🔴"
        # 로그에 청산가 정보도 살짝 표시
        liq_info = f" | Liq: ${params['liq_price']:.1f}"
        self.logger.info(f"\n{action_icon} [ENTRY] {timestamp} | {params['action']} @ ${params['entry_price']:.2f} | Qty: {params['quantity']:.4f}{liq_info}")

    def _check_exit(self, current_bar):
        """청산 확인 (우선순위: 강제청산 > 타임컷 > 손절/익절)"""
        pos = self.position
        exit_price = None
        exit_reason = ""
        
        current_time = current_bar.name
        high = current_bar['high']
        low = current_bar['low']
        open_price = current_bar['open']
        current_atr = current_bar.get('atr', 0)

        # ---------------------------------------------------------
        # [NEW] 1. 청산가 거리 측정 (위험도 기록)
        # ---------------------------------------------------------
        if pos['type'] == "BUY":
            # 롱: 저가가 청산가에 얼마나 근접했나?
            dist_percent = ((low - pos['liq_price']) / pos['entry_price']) * 100
        else:
            # 숏: 고가가 청산가에 얼마나 근접했나?
            dist_percent = ((pos['liq_price'] - high) / pos['entry_price']) * 100
        
        # 최소 거리 갱신 (더 위험했던 순간을 기록)
        if dist_percent < pos['min_liq_dist_pct']:
            pos['min_liq_dist_pct'] = dist_percent

        # ---------------------------------------------------------
        # [NEW] 2. 강제 청산 (Liquidation) 체크 - 최우선 순위
        # ---------------------------------------------------------
        is_liquidated = False
        if pos['type'] == "BUY":
            if low <= pos['liq_price']:
                is_liquidated = True
        else: # SELL
            if high >= pos['liq_price']:
                is_liquidated = True
                
        if is_liquidated:
            exit_price = pos['liq_price']
            exit_reason = "☠️ LIQUIDATION"
            self._execute_exit(exit_price, exit_reason, current_time)
            return # 강제 청산 당했으면 뒤에 로직은 의미 없음

        # ---------------------------------------------------------
        # 3. 타임 컷 (Time Cut)
        # ---------------------------------------------------------
        if pos['entry_time'] != current_time:
            exit_price = open_price
            exit_reason = "TIME_CUT (Next Open)"
            self._execute_exit(exit_price, exit_reason, current_time)
            return

        # ---------------------------------------------------------
        # 4. 트레일링 스탑 업데이트
        # ---------------------------------------------------------
        if pos.get('trailing_mult') and current_atr > 0:
            if pos['type'] == "BUY":
                new_sl = high - (current_atr * pos['trailing_mult'])
                if new_sl > pos['stop_loss']:
                    pos['stop_loss'] = new_sl
            elif pos['type'] == "SELL":
                new_sl = low + (current_atr * pos['trailing_mult'])
                if new_sl < pos['stop_loss']:
                    pos['stop_loss'] = new_sl

        # ---------------------------------------------------------
        # 5. 장중 손절 (Stop Loss)
        # ---------------------------------------------------------
        if pos['type'] == "BUY":
            if low <= pos['stop_loss']:
                exit_price = pos['stop_loss']
                exit_reason = "STOP_LOSS"
        elif pos['type'] == "SELL":
            if high >= pos['stop_loss']:
                exit_price = pos['stop_loss']
                exit_reason = "STOP_LOSS"

        if exit_price:
            self._execute_exit(exit_price, exit_reason, current_time)

    def _execute_exit(self, price, reason, timestamp):
        """청산 실행 및 로그 저장"""
        pos = self.position
        
        # PnL 계산
        if pos['type'] == "BUY":
            gross_pnl = (price - pos['entry_price']) * pos['quantity']
        else:
            gross_pnl = (pos['entry_price'] - price) * pos['quantity']
            
        exit_value = price * pos['quantity']
        fee = exit_value * (self.fee_rate + self.slippage)
        
        net_pnl = gross_pnl - fee
        self.balance += net_pnl
        
        # 로그 저장
        self.trade_log.append({
            "entry_time": pos['entry_time'],
            "exit_time": timestamp,
            "type": pos['type'],
            "pnl": round(net_pnl, 2),
            "reason": reason,
            "balance": round(self.balance, 2),
            
            # [수정됨] 누락되었던 핵심 정보 추가!
            "entry_price": pos['entry_price'],
            "exit_price": price,
            "quantity": pos['quantity'],
            
            # 위험도 정보
            "liq_price": pos['liq_price'],
            "min_dist_pct": round(pos['min_liq_dist_pct'], 2)
        })
        
        pnl_icon = "💰" if net_pnl > 0 else "💸"
        if "LIQUIDATION" in reason: pnl_icon = "☠️"
        
        self.logger.info(f"{pnl_icon} [EXIT] {timestamp} | {reason} | PnL: ${net_pnl:.2f} | Min Dist: {pos['min_liq_dist_pct']:.2f}%")
        
        self.position = None

    def _record_equity(self, current_bar):
        """자산 가치 기록 (Mark to Market)"""
        equity = self.balance
        if self.position:
            current_price = current_bar['close']
            pos = self.position
            if pos['type'] == "BUY":
                unrealized_pnl = (current_price - pos['entry_price']) * pos['quantity']
            else:
                unrealized_pnl = (pos['entry_price'] - current_price) * pos['quantity']
            equity += unrealized_pnl
            
        self.equity_curve.append({"timestamp": current_bar.name, "equity": equity})

    def _generate_report(self):
        """최종 리포트 생성 (위험도 분석 포함)"""
        # 로그 저장
        df_decisions = pd.DataFrame(self.decision_log)
        decision_path = os.path.join(Config.SAVE_DIR, "decision_log.csv")
        df_decisions.to_csv(decision_path, index=False)
        self.logger.info(f"\n[Bot] 📝 Decision Log saved to {decision_path}")

        if not self.trade_log:
            self.logger.info("\n[Backtest Result] No trades executed.")
            # [수정] 거래가 없어도 빈 결과를 반환하도록 수정
            return {
                "roi_pct": 0, "mdd_pct": 0, "total_trades": 0, 
                "win_rate_pct": 0, "liquidations": 0
            }

        df_trades = pd.DataFrame(self.trade_log)
        
        # 통계 계산
        total_trades = len(df_trades)
        wins = len(df_trades[df_trades['pnl'] > 0])
        win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
        
        # [NEW] 위험도 분석
        # 청산가와의 거리가 5% 이내였던 적이 있는 매매 (Near Death)
        near_death_trades = len(df_trades[df_trades['min_dist_pct'] <= 5.0])
        liquidated_trades = len(df_trades[df_trades['reason'].str.contains("LIQUIDATION")])

        # MDD & ROI
        equity_series = pd.DataFrame(self.equity_curve).set_index('timestamp')['equity']
        rolling_max = equity_series.cummax()
        drawdown = (equity_series - rolling_max) / rolling_max
        mdd = drawdown.min() * 100
        roi = ((self.balance - Config.INITIAL_BALANCE) / Config.INITIAL_BALANCE) * 100
        
        # 결과 출력
        self.logger.info(f"\n{'='*40}")
        self.logger.info(f"📊 BACKTEST FINAL REPORT")
        self.logger.info(f"{'='*40}")
        self.logger.info(f"Initial Balance: ${Config.INITIAL_BALANCE:,.2f}")
        self.logger.info(f"Final Balance:   ${self.balance:,.2f}")
        self.logger.info(f"Return (ROI):    {roi:.2f}%")
        self.logger.info(f"MDD:             {mdd:.2f}%")
        self.logger.info(f"Total Trades:    {total_trades}")
        self.logger.info(f"Win Rate:        {win_rate:.1f}%")
        self.logger.info(f"{'-'*40}")
        self.logger.info(f"☠️ Liquidations:  {liquidated_trades} trades")
        self.logger.info(f"⚠️ Near Death (<5%): {near_death_trades} trades")
        self.logger.info(f"{'='*40}")
        
        # 파일 저장
        df_trades.to_csv(os.path.join(Config.SAVE_DIR, "trade_log.csv"), index=False)
        equity_series.to_csv(os.path.join(Config.SAVE_DIR, "equity_curve.csv"))
        
        # [핵심 수정] 결과를 딕셔너리로 묶어서 반환
        return {
            "roi_pct": roi,
            "mdd_pct": mdd,
            "total_trades": total_trades,
            "win_rate_pct": win_rate,
            "liquidations": liquidated_trades
        }