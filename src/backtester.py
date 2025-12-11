import pandas as pd
import sys
import os
import traceback
from tqdm import tqdm
from datetime import timedelta

# 상위 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from config.settings import Config
from src.logger import setup_logger

class Backtester:
    """
    [Step 2 Modified]
    - 종목별 결과 파일 분리 (trade_log_SYMBOL.csv)
    - 1h 데이터 기반, Intra-bar 손절, 펀딩비 계산 포함
    - 격리 마진(Isolated Margin) 로직 정밀 구현: 청산 시 최대 손실을 증거금으로 제한
    """
    def __init__(self, df: pd.DataFrame, strategy, risk_manager, symbol: str):
        self.full_df = df
        self.strategy = strategy
        self.risk_manager = risk_manager
        self.symbol = symbol  # [New] 현재 백테스팅 중인 심볼 (예: BTC/USDT)
        
        # 로거 초기화
        self.logger = setup_logger()
        
        # 상태 변수
        self.balance = Config.INITIAL_BALANCE # (main.py에서 할당된 금액으로 덮어씌워짐)
        self.position = None 
        self.trade_log = []     
        self.equity_curve = []  
        self.decision_log = []  
        
        # 비용 설정
        self.fee_rate = Config.TRADING_FEE_RATE
        self.slippage = Config.SLIPPAGE
        self.funding_rate_8h = Config.FUNDING_RATE_8H 

    def run(self):
        """백테스팅 메인 루프 (Hourly)"""
        self.full_df = self.strategy.calculate_indicators(self.full_df)

        if len(self.full_df) < 100:
             self.logger.error(f"❌ [{self.symbol}] Not enough data.")
             return None

        needed_bars = Config.TEST_DAYS * 24
        start_index = max(0, len(self.full_df) - needed_bars)
        
        self.logger.info(f"\n[Backtester] {self.symbol} | 1h Data | Last {Config.TEST_DAYS} days")
        self.logger.info(f"Start Balance: ${self.balance:,.2f}")
        
        show_progress = False if Config.LOG_LEVEL == "WARNING" else True
        
        for row in tqdm(self.full_df.iloc[start_index:].itertuples(), total=len(self.full_df)-start_index, disable=not show_progress, desc=f"{self.symbol}"):
            # [Optimization] itertuples() 사용으로 객체 생성 오버헤드 제거
            # row는 NamedTuple (Index, open, high, low, close, ...)
            
            current_time = row.Index # 인덱스는 Index 속성으로 접근
            
            if self.position:
                self._apply_funding_fee(current_time)
                self._check_exit(row)
            
            if not self.position:
                # [Optimization] 선계산된 시그널 확인 (함수 호출 비용 제거)
                # row.buy_signal이 True일 때만 진입 로직 수행
                if row.buy_signal:
                    self._process_entry(row)
            
            self._record_equity(row)

        return self._generate_report()

    def _process_entry(self, row):
        # [Optimization] generate_signal 호출 제거하고 row 데이터 직접 사용
        # 이미 buy_signal이 True인 상태에서 들어옴
        
        action = "BUY"
        # long_target 등은 이미 row에 있음
        entry_price = row.long_target
        
        # 손절가 계산 (전략 로직 인라인 or row에서 가져오기)
        # 전략 객체의 파라미터 필요 (sl_multiplier)
        daily_atr = row.prev_atr
        stop_loss = entry_price - (daily_atr * self.strategy.sl_multiplier)
        
        params = self.risk_manager.calculate_entry_params(
            balance=self.balance,
            entry_price=entry_price,
            stop_loss=stop_loss,
            action=action
        )
        
        if params:
            # Trailing Stop Multiplier도 전략에서 가져옴
            params['trailing_stop_multiplier'] = self.strategy.trailing_mult
            self._execute_entry(params, row.Index)
            
            # Intra-bar Loss Check (속성 접근)
            if row.low <= params['stop_loss']:
                self.logger.warning(f"⚠️ [{self.symbol}] Intra-bar Loss immediately!")
                self._execute_exit(
                    price=params['stop_loss'],
                    reason="STOP_LOSS (Intra-bar)",
                    timestamp=row.Index
                )

    def _execute_entry(self, params, timestamp):
        """진입 주문 및 증거금 기록"""
        cost_rate = self.fee_rate + self.slippage
        entry_value = params['entry_price'] * params['quantity']
        fee = entry_value * cost_rate
        
        self.balance -= fee
        
        # [New] 격리 증거금(Margin) 계산 및 저장
        # Config.LEVERAGE는 main.py에서 해당 종목에 맞게 설정되어 있다고 가정
        isolated_margin = entry_value / Config.LEVERAGE
        
        self.position = {
            "type": params['action'],
            "entry_price": params['entry_price'],
            "quantity": params['quantity'],
            "stop_loss": params['stop_loss'],
            "entry_time": timestamp,
            "liq_price": params['liq_price'], 
            "min_liq_dist_pct": 100.0,
            "margin": isolated_margin # <--- 증거금 정보 저장
        }
        
        self.logger.info(f"\n🟢 [{self.symbol}] ENTRY {timestamp} | BUY @ ${params['entry_price']:.2f} | Margin: ${isolated_margin:.2f}")

    def _check_exit(self, row):
        pos = self.position
        current_time = row.Index
        
        curr_low = row.low
        curr_open = row.open

        dist_percent = ((curr_low - pos['liq_price']) / pos['entry_price']) * 100
        if dist_percent < pos['min_liq_dist_pct']:
            pos['min_liq_dist_pct'] = dist_percent

        if curr_low <= pos['liq_price']:
            self._execute_exit(pos['liq_price'], "☠️ LIQUIDATION", current_time)
            return 

        if curr_low <= pos['stop_loss']:
            self._execute_exit(pos['stop_loss'], "STOP_LOSS", current_time)
            return

        if pos['entry_time'].date() < current_time.date():
            self._execute_exit(curr_open, "TIME_CUT (New Day)", current_time)
            return

    def _apply_funding_fee(self, current_time):
        """펀딩비 차감"""
        if current_time.hour in [0, 8, 16] and current_time.minute == 0:
            pos = self.position
            position_value = pos['entry_price'] * pos['quantity']
            fee = position_value * self.funding_rate_8h
            self.balance -= fee

    def _execute_exit(self, price, reason, timestamp):
        """청산 실행 (격리 마진 보호 로직 적용 + 로그 정보 보강)"""
        pos = self.position
        
        # 1. 기본 PnL 계산
        gross_pnl = (price - pos['entry_price']) * pos['quantity']
        exit_value = price * pos['quantity']
        fee = exit_value * (self.fee_rate + self.slippage)
        
        # 2. 강제 청산 시 손실 제한 (Isolated Margin Protection)
        if "LIQUIDATION" in reason:
            net_pnl = -pos['margin'] 
            self.logger.warning(f"🛡️ [{self.symbol}] Isolated Margin Liquidated. Max Loss capped at -${pos['margin']:.2f}")
        else:
            net_pnl = gross_pnl - fee

        self.balance += net_pnl
        
        # [수정] 누락되었던 entry_price, exit_price, quantity 다시 추가!
        self.trade_log.append({
            "entry_time": pos['entry_time'],
            "exit_time": timestamp,
            "type": pos['type'],
            "pnl": round(net_pnl, 2),
            "reason": reason,
            "balance": round(self.balance, 2),
            "min_dist_pct": round(pos['min_liq_dist_pct'], 2),
            
            # --- 여기가 복구된 부분입니다 ---
            "entry_price": pos['entry_price'],
            "exit_price": price,
            "quantity": pos['quantity']
            # ---------------------------
        })
        
        icon = "💰" if net_pnl > 0 else "💸"
        if "LIQUIDATION" in reason: icon = "☠️"
        
        self.logger.info(f"{icon} [{self.symbol}] EXIT {timestamp} | {reason} | PnL: ${net_pnl:.2f}")
        self.position = None


    def _record_equity(self, row):
        equity = self.balance
        if self.position:
            current_price = row.close
            unrealized = (current_price - self.position['entry_price']) * self.position['quantity']
            equity += unrealized
        self.equity_curve.append({"timestamp": row.Index, "equity": equity})


    def _generate_report(self):
        """최종 리포트 생성 (ROI 반환 누락 수정)"""
        
        # 거래가 아예 없었을 경우
        if not self.trade_log:
            return {
                "roi_pct": 0.0, 
                "mdd_pct": 0.0, 
                "total_trades": 0, 
                "win_rate_pct": 0.0, 
                "liquidations": 0,
                "min_safety_margin": 100.0
            }

        df_trades = pd.DataFrame(self.trade_log)
        df_equity = pd.DataFrame(self.equity_curve).set_index('timestamp')
        
        # 1. ROI 계산 (누락되었던 부분)
        # 봇 내부 기준 ROI (record_test.py 용)
        roi = ((self.balance - Config.INITIAL_BALANCE) / Config.INITIAL_BALANCE) * 100
        
        # 2. MDD 계산
        if not df_equity.empty:
            peak = df_equity['equity'].cummax()
            drawdown = (df_equity['equity'] - peak) / peak
            mdd = drawdown.min() * 100
        else:
            mdd = 0.0
        
        # 3. 통계 지표
        total_trades = len(df_trades)
        wins = len(df_trades[df_trades['pnl'] > 0])
        win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
        liquidations = len(df_trades[df_trades['reason'].str.contains("LIQUIDATION")])
        min_safety_margin = df_trades['min_dist_pct'].min() if 'min_dist_pct' in df_trades.columns else 0.0

        # 파일 저장 (심볼명 포함)
        safe_symbol = self.symbol.replace("/", "_")
        trade_file = os.path.join(Config.SAVE_DIR, f"trade_log_{safe_symbol}.csv")
        equity_file = os.path.join(Config.SAVE_DIR, f"equity_curve_{safe_symbol}.csv")
        
        df_trades.to_csv(trade_file, index=False)
        df_equity.to_csv(equity_file)

        self.logger.info(f"\n📊 [{self.symbol} Result] ROI: {roi:.2f}% | WinRate: {win_rate:.1f}% | MDD: {mdd:.2f}%")
        
        # [수정 완료] roi_pct 키 추가
        return {
            "roi_pct": roi,               # <--- 여기 추가되었습니다!
            "mdd_pct": mdd,
            "total_trades": total_trades,
            "win_rate_pct": win_rate,
            "liquidations": liquidations,
            "min_safety_margin": min_safety_margin
        }