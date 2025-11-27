import sys
import os
import shutil
import backtrader as bt
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict # [NEW] PnL 추적용

# 프로젝트 루트 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import CONFIG, DATA_DIR
from alpha_layer.brain import Brain
from data_layer.data_handler import MultiSymbolLoader

# --------------------------------------------------------- 
# 1. 데이터 피드 및 공통 클래스 (기존 유지)
# --------------------------------------------------------- 
class HMMData(bt.feeds.PandasData):
    lines = ('regime',)
    params = (
        ('regime', 'Regime'),
        ('datetime', None),
        ('open', 'open'),
        ('high', 'high'),
        ('low', 'low'),
        ('close', 'close'),
        ('volume', 'volume'),
        ('openinterest', -1),
    )

# [수정] 슬리피지 로직 제거 및 순수 수수료 클래스로 변경
class FuturesComm(bt.CommInfoBase):
    params = (
        ('stocklike', False), 
        ('commtype', bt.CommInfoBase.COMM_PERC),
        ('perc', 0.0005), 
        ('leverage', 1.0)
    )

    def _getcommission(self, size, price, pseudoexec=None, **kwargs):
        # 슬리피지 비용 가산 로직 제거 (Broker 레벨에서 처리됨)
        return super()._getcommission(size, price, pseudoexec, **kwargs)

    def get_margin(self, price): 
        return price / self.p.leverage

# --------------------------------------------------------- 
# [EXISTING] 개별 코인용 전략 (데이터 정밀도 및 안전장치 강화판)
# --------------------------------------------------------- 
class HMM_Pro_Strategy_V5(bt.Strategy):
    params = (
        ('bull_id', None), ('bear_id', None),
        ('leverage', 2.0),
        ('max_positions', 3),
        ('atr_period', 14), ('rsi_period', 14), ('ema_period', 20),
        ('rsi_buy_upper', 65), ('rsi_buy_lower', 30),
        ('rsi_sell_lower', 35), ('rsi_sell_upper', 70),
        ('stop_atr', 2.0), ('trail_trigger', 2.0), ('trail_dist', 2.0),
        ('cooldown_bars', 2),
        ('output_dir', None),
        ('funding_rate', 0.0001) 
    )
    def __init__(self):
        self.regime = self.data.regime
        self.atr = bt.indicators.ATR(self.data, period=self.p.atr_period)
        self.rsi = bt.indicators.RSI(self.data, period=self.p.rsi_period)
        self.ema = bt.indicators.EMA(self.data.close, period=self.p.ema_period)
        
        self.order = None
        self.entry_price = 0.0
        self.stop_price = 0.0
        self.cooldown = 0
        
        # [데이터 추적 고도화]
        self.entry_metadata = {} # 진입 시점의 정보 저장 (Size, Regime 등)
        self.exit_reason = "Manual/End" 
        
        self.trade_log = []   # 상세 매매 기록 (CSV 저장용)
        self.daily_stats = [] # 일별 통계
        self.csv_path = ""

    def notify_order(self, order):
        if order.status in [order.Completed]:
            executed_price = order.executed.price
            executed_size = abs(order.executed.size)

            if order.isbuy():
                if self.position.size > 0: # Long Entry
                    self.entry_price = executed_price
                    self.stop_price = self.entry_price - (self.atr[0] * self.p.stop_atr)
                    # [핵심] 진입 시점 데이터 기록
                    self.entry_metadata = {
                        'size': executed_size,
                        'entry_time': self.data.datetime.datetime(0),
                        'regime': 'Bull' if int(self.regime[0]) == self.p.bull_id else 'Bear'
                    }
                # Exit은 별도 처리 안 함 (notify_trade에서 처리)
                
            elif order.issell():
                if self.position.size < 0: # Short Entry
                    self.entry_price = executed_price
                    self.stop_price = self.entry_price + (self.atr[0] * self.p.stop_atr)
                    # [핵심] 진입 시점 데이터 기록
                    self.entry_metadata = {
                        'size': executed_size,
                        'entry_time': self.data.datetime.datetime(0),
                        'regime': 'Bull' if int(self.regime[0]) == self.p.bull_id else 'Bear'
                    }
            
            self.order = None
            self.cooldown = 0
            
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed: return
        
        # 1. 데이터 복원 (Entry Metadata 사용)
        # trade.size는 0이므로, 진입 시 저장해둔 size 사용
        real_size = self.entry_metadata.get('size', 0.0)
        
        # 메타데이터가 없을 경우(예외) history에서 복구 시도
        if real_size == 0 and trade.history:
            real_size = abs(trade.history[0].event.size)
            
        entry_time = self.entry_metadata.get('entry_time', bt.num2date(trade.dtopen))
        entry_regime = self.entry_metadata.get('regime', 'Unknown')
        
        # 2. 안전한 계산 (ZeroDivision 방지)
        pnl_net = trade.pnlcomm
        pnl_gross = trade.pnl
        
        # Exit Price 역산
        final_exit_price = trade.price
        if real_size > 0:
            final_exit_price = trade.price + (pnl_gross / real_size)
            
        # ROI 계산
        safe_leverage = self.p.leverage if self.p.leverage > 0 else 1.0
        entry_val = (trade.price * real_size) / safe_leverage
        
        roi_pct = 0.0
        if entry_val != 0:
            roi_pct = (pnl_net / entry_val) * 100

        # Duration
        exit_time = bt.num2date(trade.dtclose)
        duration = exit_time - entry_time

        # 3. 로그 적재 (Master Log 포맷)
        self.trade_log.append({
            'Symbol': self.data._name,
            'Type': 'Long' if trade.long else 'Short',
            'Entry_Time': entry_time,
            'Exit_Time': exit_time,
            'Duration_Hours': duration.total_seconds() / 3600,
            'Entry_Price': trade.price,
            'Exit_Price': final_exit_price,
            'Size': real_size,
            'PnL': pnl_net,
            'Fee': trade.commission,
            'ROI': roi_pct,
            'Entry_Regime': entry_regime,
            'Exit_Reason': self.exit_reason
        })
        
        # 청산 후 이유 초기화
        self.exit_reason = "Manual/End" 

    def next(self):
        # 1. 일별 통계 기록
        self.daily_stats.append({
            'Date': self.data.datetime.datetime(0),
            'Open': self.data.open[0],
            'High': self.data.high[0],
            'Low': self.data.low[0],
            'Close': self.data.close[0],
            'RSI': self.rsi[0],
            'ATR': self.atr[0],
            'Regime': int(self.regime[0]),
            'Position_Size': self.position.size,
            'Portfolio_Value': self.broker.getvalue()
        })
        
        # Funding Fee Simulation
        dt = self.data.datetime.datetime(0)
        if dt.minute == 0 and dt.hour in [0, 8, 16]:
            if self.position.size != 0:
                pos_value = abs(self.position.size) * self.data.close[0]
                funding_cost = pos_value * self.p.funding_rate
                self.broker.add_cash(-funding_cost)

        # 3. 매매 로직
        if self.order: return
        close = self.data.close[0]
        regime = int(self.regime[0])
        atr = self.atr[0]
        rsi = self.rsi[0]
        ema = self.ema[0]
        if self.cooldown > 0: self.cooldown -= 1

        # [청산 로직]
        if self.position.size != 0:
            should_close = False
            reason = ""
            
            if self.position.size > 0: # Long
                if self.data.low[0] < self.stop_price: 
                    should_close = True; reason = "StopLoss"
                elif close > self.entry_price + (atr * self.p.trail_trigger):
                    new_stop = close - (atr * self.p.trail_dist)
                    if new_stop > self.stop_price: self.stop_price = new_stop
                elif regime == self.p.bear_id: 
                    should_close = True; reason = "RegimeChange(Bear)"
                    
            elif self.position.size < 0: # Short
                if self.data.high[0] > self.stop_price: 
                    should_close = True; reason = "StopLoss"
                elif close < self.entry_price - (atr * self.p.trail_trigger):
                    new_stop = close + (atr * self.p.trail_dist)
                    if new_stop < self.stop_price: self.stop_price = new_stop
                elif regime == self.p.bull_id: 
                    should_close = True; reason = "RegimeChange(Bull)"
            
            if should_close:
                self.exit_reason = reason # 청산 사유 기록
                self.close()
                self.cooldown = self.p.cooldown_bars
                return
        
        # [진입 로직]
        elif self.position.size == 0 and self.cooldown == 0:
            cash = self.broker.get_cash()
            if cash <= 0: return
            
            total_equity = self.broker.getvalue()
            
            # [FIX] 안전한 나눗셈
            max_pos = self.p.max_positions if self.p.max_positions > 0 else 1
            allocation = total_equity / max_pos
            target_value = allocation * 0.95 * self.p.leverage
            
            size = 0
            if close > 0:
                size = target_value / close
            
            if size < 0.000001: return 

            if regime == self.p.bull_id:
                if close > ema and self.p.rsi_buy_lower < rsi < self.p.rsi_buy_upper:
                    self.order = self.buy(size=size)
            elif regime == self.p.bear_id:
                if close < ema and self.p.rsi_sell_lower < rsi < self.p.rsi_sell_upper:
                    self.order = self.sell(size=size)

    def stop(self):
        save_dir = self.p.output_dir
        if not save_dir: return

        # 1. Master Trade Log (통합 거래 대장 - 상세 분석용)
        if self.trade_records:
            df_trades = pd.DataFrame(self.trade_records)
            df_trades.sort_values(by='Entry_Time', inplace=True)
            df_trades.to_csv(os.path.join(save_dir, 'Master_Trade_Log.csv'), index=False)
            
        # 2. Daily Portfolio Stats (전체 포트폴리오 자산 흐름)
        if self.daily_stats:
            df_stats = pd.DataFrame(self.daily_stats)
            df_stats.drop_duplicates(subset=['Date'], keep='last', inplace=True)
            df_stats.to_csv(os.path.join(save_dir, 'Daily_Portfolio_Stats.csv'), index=False)

        # 3. [수정됨] Asset Equity Curves (개별 코인 자산 흐름 통합 저장)
        # 기존: 코인별로 BT_...csv 파일 수십 개 생성 (지저분함)
        # 변경: Asset_Equity_Curves.csv 하나에 날짜별로 컬럼을 만들어 저장
        try:
            combined_df = pd.DataFrame()
            
            for name, history in self.pnl_history.items():
                if not history: continue
                
                # 개별 히스토리 DF 생성
                df_coin = pd.DataFrame(history)
                df_coin['Date'] = pd.to_datetime(df_coin['Date'])
                df_coin.set_index('Date', inplace=True)
                
                # 컬럼명 변경 (Portfolio_Value -> 코인심볼)
                clean_name = name.replace('/', '')
                df_coin.rename(columns={'Portfolio_Value': clean_name}, inplace=True)
                
                # 중복 제거 (하루에 여러 틱이 있을 경우 마지막 값 사용)
                df_coin = df_coin[~df_coin.index.duplicated(keep='last')]
                
                # 통합 DF에 병합
                if combined_df.empty:
                    combined_df = df_coin
                else:
                    combined_df = combined_df.join(df_coin, how='outer')
            
            if not combined_df.empty:
                combined_df.sort_index(inplace=True)
                combined_df.fillna(method='ffill', inplace=True) # 앞의 값으로 채우기
                combined_df.fillna(0, inplace=True) # 앞의 값 없으면 0
                combined_df.reset_index(inplace=True)
                
                combined_df.to_csv(os.path.join(save_dir, 'Asset_Equity_Curves.csv'), index=False)
                
        except Exception as e:
            print(f"⚠️ 자산 곡선 병합 중 오류: {e}")

# --------------------------------------------------------- 
# [NEW] 리얼 포트폴리오 전략 (Size 0 버그 수정판)
# --------------------------------------------------------- 
class RealPortfolioStrategy(bt.Strategy):
    params = (
        ('regime_map', {}),
        ('max_positions', 3),
        ('leverage', 2.0),
        ('config', {}),
        ('output_dir', None)
    )

    def __init__(self):
        self.inds = {}
        self.orders = {}
        self.cooldowns = {} 
        self.stops = {}
        
        # [데이터 추적용]
        # entry_metadata에 'size'를 저장하여 청산 시 참조합니다.
        self.entry_metadata = {} 
        self.exit_reasons = {}
        
        self.trade_records = []
        self.daily_stats = []
        self.pnl_history = defaultdict(list)
        self.realized_pnl = defaultdict(float)

        for d in self.datas:
            name = d._name
            self.orders[name] = None
            self.cooldowns[name] = 0
            self.stops[name] = 0.0
            self.entry_metadata[name] = {}
            self.exit_reasons[name] = "Manual/End"
            
            cfg = self.p.config
            atr = bt.indicators.ATR(d, period=int(cfg.get('ATR_PERIOD', 14)))
            rsi = bt.indicators.RSI(d, period=int(cfg.get('RSI_PERIOD', 14)))
            ema = bt.indicators.EMA(d.close, period=int(cfg.get('EMA_PERIOD', 20)))
            
            self.inds[name] = {'atr': atr, 'rsi': rsi, 'ema': ema}

    def notify_order(self, order):
        name = order.data._name
        if order.status in [order.Completed]:
            # [진입] Buy or Sell Entry
            # 포지션이 생긴 시점의 정보를 기록합니다.
            if (order.isbuy() and self.getposition(order.data).size > 0) or \
               (order.issell() and self.getposition(order.data).size < 0):
                
                self.entry_metadata[name] = {
                    'entry_price': order.executed.price,
                    'entry_time': order.data.datetime.datetime(0),
                    'size': abs(order.executed.size), # [핵심] 진입 사이즈 기록!
                    'regime': 'Bull' if self.inds[name].get('curr_regime') == self.p.regime_map[name]['bull'] else 'Bear'
                }
                
                atr_val = self.inds[name]['atr'][0]
                stop_atr = float(self.p.config.get('STOP_LOSS_ATR', 2.0))
                if order.isbuy():
                    self.stops[name] = order.executed.price - (atr_val * stop_atr)
                else:
                    self.stops[name] = order.executed.price + (atr_val * stop_atr)

            self.orders[name] = None
            self.cooldowns[name] = 0
            
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.orders[name] = None

    def notify_trade(self, trade):
        if not trade.isclosed: return
        name = trade.data._name
        
        pnl_net = trade.pnlcomm
        self.realized_pnl[name] += pnl_net
        
        # 저장해둔 메타데이터 불러오기
        meta = self.entry_metadata.get(name, {})
        entry_time = meta.get('entry_time', bt.num2date(trade.dtopen))
        
        # [핵심 수정] trade.size는 0이므로, 진입 시 저장한 size를 사용
        real_size = meta.get('size', 0.0)
        
        # 만약 메타데이터가 꼬여서 size가 없다면, history에서 추론 (안전장치)
        if real_size == 0 and trade.history:
            real_size = abs(trade.history[0].event.size)

        exit_time = bt.num2date(trade.dtclose)
        duration = exit_time - entry_time
        
        # ROI 계산 (레버리지 반영)
        # real_size를 사용하므로 0으로 나누는 일 없음
        safe_leverage = self.p.leverage if self.p.leverage > 0 else 1.0
        entry_val = (trade.price * real_size) / safe_leverage
        
        roi_pct = 0.0
        if entry_val != 0:
            roi_pct = (pnl_net / entry_val) * 100

        # Exit Price 역산
        exit_price_approx = trade.price
        if real_size != 0:
            # PnL = (Exit - Entry) * Size  => Exit = Entry + (PnL/Size) (Long 기준)
            # Short도 PnL 부호가 반대라 수식 동일
            exit_price_approx = trade.price + (trade.pnl / real_size)

        self.trade_records.append({
            'Symbol': name,
            'Side': 'Long' if trade.long else 'Short',
            'Entry_Time': entry_time,
            'Exit_Time': exit_time,
            'Duration_Hours': duration.total_seconds() / 3600,
            'Entry_Price': trade.price,
            'Exit_Price': exit_price_approx,
            'Size': real_size, # 0.0이 아닌 실제 사이즈 기록
            'PnL_Net': pnl_net,
            'Fee': trade.commission,
            'ROI_Pct': roi_pct,
            'Entry_Regime': meta.get('regime', 'Unknown'),
            'Exit_Reason': self.exit_reasons.get(name, 'Unknown')
        })

    def next(self):
        dt = self.datas[0].datetime.datetime(0)
        open_positions_count = sum(1 for d in self.datas if self.getposition(d).size != 0)
        
        self.daily_stats.append({
            'Date': dt,
            'Total_Equity': self.broker.getvalue(),
            'Cash': self.broker.getcash(),
            'Open_Positions': open_positions_count,
            'Leverage': self.p.leverage
        })

        for d in self.datas:
            name = d._name
            pos = self.getposition(d)
            unrealized = 0.0
            if pos.size != 0:
                unrealized = pos.size * (d.close[0] - pos.price)
            total_contribution = self.realized_pnl[name] + unrealized
            self.pnl_history[name].append({'Date': dt, 'Portfolio_Value': total_contribution})

        max_pos = int(self.p.config.get('MAX_OPEN_POSITIONS', 3))
        if max_pos <= 0: max_pos = 1
        
        cfg = self.p.config
        rsi_buy_low = float(cfg.get('RSI_BUY_LOWER', 30))
        rsi_buy_high = float(cfg.get('RSI_BUY_UPPER', 65))
        rsi_sell_low = float(cfg.get('RSI_SELL_LOWER', 35))
        rsi_sell_high = float(cfg.get('RSI_SELL_UPPER', 70))
        trail_trigger = float(cfg.get('TRAIL_TRIGGER_ATR', 2.0))
        trail_dist = float(cfg.get('TRAIL_DIST_ATR', 2.0))

        for d in self.datas:
            name = d._name
            pos = self.getposition(d)
            
            regime_info = self.p.regime_map.get(name)
            if not regime_info: continue
            
            curr_regime = int(d.regime[0])
            self.inds[name]['curr_regime'] = curr_regime
            
            inds = self.inds[name]
            atr = inds['atr'][0]
            rsi = inds['rsi'][0]
            ema = inds['ema'][0]
            close = d.close[0]
            
            if self.cooldowns[name] > 0: self.cooldowns[name] -= 1

            if pos.size != 0:
                stop_price = self.stops[name]
                should_close = False
                reason = ""

                if pos.size > 0: # Long
                    if d.low[0] < stop_price: should_close = True; reason = "StopLoss"
                    elif close > self.entry_metadata[name].get('entry_price', 0) + (atr * trail_trigger):
                        new_stop = close - (atr * trail_dist)
                        if new_stop > stop_price: self.stops[name] = new_stop
                    elif curr_regime == regime_info['bear']:
                        should_close = True; reason = "RegimeChange(Bear)"
                else: # Short
                    if d.high[0] > stop_price: should_close = True; reason = "StopLoss"
                    elif close < self.entry_metadata[name].get('entry_price', 0) - (atr * trail_trigger):
                        new_stop = close + (atr * trail_dist)
                        if new_stop < stop_price: self.stops[name] = new_stop
                    elif curr_regime == regime_info['bull']:
                        should_close = True; reason = "RegimeChange(Bull)"

                if should_close:
                    self.exit_reasons[name] = reason
                    self.close(data=d)
                    self.cooldowns[name] = 2

            elif self.orders[name] is None and self.cooldowns[name] == 0:
                if open_positions_count >= max_pos: continue
                
                allocation = self.broker.getvalue() / max_pos
                target_amt = allocation * 0.95 * self.p.leverage
                
                size = 0
                if close > 0:
                    size = target_amt / close
                
                if size <= 0: continue

                if curr_regime == regime_info['bull']:
                    if close > ema and rsi_buy_low < rsi < rsi_buy_high:
                        self.orders[name] = self.buy(data=d, size=size)
                        open_positions_count += 1
                elif curr_regime == regime_info['bear']:
                    if close < ema and rsi_sell_low < rsi < rsi_sell_high:
                        self.orders[name] = self.sell(data=d, size=size)
                        open_positions_count += 1

    def stop(self):
        save_dir = self.p.output_dir
        if not save_dir: return

        if self.trade_records:
            df_trades = pd.DataFrame(self.trade_records)
            df_trades.sort_values(by='Entry_Time', inplace=True)
            df_trades.to_csv(os.path.join(save_dir, 'Master_Trade_Log.csv'), index=False)
            
        if self.daily_stats:
            df_stats = pd.DataFrame(self.daily_stats)
            df_stats.drop_duplicates(subset=['Date'], keep='last', inplace=True)
            df_stats.to_csv(os.path.join(save_dir, 'Daily_Portfolio_Stats.csv'), index=False)

        for name, history in self.pnl_history.items():
            if history:
                df_pnl = pd.DataFrame(history)
                clean_name = name.replace('/', '')
                ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"BT_{ts}_{clean_name}.csv"
                df_pnl.to_csv(os.path.join(save_dir, filename), index=False)


# --------------------------------------------------------- 
# [EXISTING] 기존 실행 함수들 (유지)
# --------------------------------------------------------- 
def run_walk_forward(symbol, initial_cash=10000.0, train_days=365, test_days=30, update_data=False, log_func=None, dynamic_config=None, output_dir=None):
    def log(msg):
        if log_func: log_func(msg)
        else: print(msg)

    # 1. 설정 로드
    cfg = dynamic_config if dynamic_config else CONFIG

    # 2. 데이터 업데이트 (요청 시)
    if update_data:
        log(f"📥 [{symbol}] 최신 데이터 다운로드 및 갱신 중...")
        try:
            loader = MultiSymbolLoader()
            required_days = train_days + test_days + 100
            fetch_days = max(cfg.get('FETCH_DAYS', 1500), required_days)
            df, path = loader.fetch_ohlcv(symbol, days=fetch_days)
            if not df.empty:
                df = loader.add_features(df)
                df.to_csv(path)
                log(f"✅ [{symbol}] 데이터 저장 완료")
            else:
                log(f"⚠️ [{symbol}] 데이터 수신 실패")
        except Exception as e:
            log(f"❌ 데이터 갱신 실패: {e}")
            return {"error": str(e)}

    # 3. 데이터 로드 및 분할
    brain = Brain()
    df_full, _ = brain.load_data(symbol)

    if df_full is None or len(df_full) < (train_days + test_days):
        log(f"❌ [{symbol}] 데이터 부족")
        return {"error": "Not enough data"}

    current_cash = initial_cash
    end_date = df_full.index[-1]
    
    test_start = end_date - timedelta(days=test_days)
    train_end = test_start - timedelta(seconds=1)
    train_start = train_end - timedelta(days=train_days)
    
    if train_start < df_full.index[0]: train_start = df_full.index[0]

    df_train = df_full.loc[train_start:train_end].copy()
    df_test = df_full.loc[test_start:end_date].copy()

    if len(df_train) < 100 or len(df_test) < 10:
        return {"error": "Data slice too small"}

    log(f"🧠 [{symbol}] AI 모델 학습 중... (Train: {len(df_train)} 캔들)")
    
    all_trade_logs = []
    equity_curve = []
    mdd = 0.0
    csv_path = ""

    try:
        # 4. 모델 학습 및 국면 식별
        model, df_train_res = brain.train_model(df_train, symbol)
        regime_map, stats = brain.identify_regimes(df_train_res)
        sorted_stats = stats.sort_values()
        bear_id = sorted_stats.index[0]
        bull_id = sorted_stats.index[-1]

        # 5. 테스트 데이터 예측
        test_features = df_test[['Log_Returns_Scaled', 'Range_Vol_Scaled', 'RSI_14_Scaled', 'OBV_Scaled']].values
        df_test['Regime'] = model.predict(test_features)
        
        # 6. Backtrader 설정
        cerebro = bt.Cerebro()
        data_feed = HMMData(dataname=df_test, name=symbol)
        cerebro.adddata(data_feed)
        
        # 레버리지 및 포지션 설정
        base_leverage = float(cfg.get('LEVERAGE', 2.0))
        if base_leverage <= 0: base_leverage = 1.0
        
        max_pos = int(cfg.get('MAX_OPEN_POSITIONS', 3))
        
        # ------------------------------------------------------------------
        # [Phase 2 핵심 변경사항] 브로커 및 수수료 설정 고도화
        # ------------------------------------------------------------------
        cerebro.broker.setcash(current_cash)

        # A. 슬리피지 설정 (비용 가산 방식이 아닌, 체결 가격 자체를 불리하게 적용)
        cerebro.broker.set_slippage_perc(
            perc=float(cfg.get('SLIPPAGE_PCT', 0.0002)),
            slip_open=True,  # 시가 진입 시에도 적용
            slip_match=True, # 슬리피지 적용된 가격으로 매칭
            slip_out=False   # 청산 시에도 적용할지 여부 (False: 보수적 관점 유지)
        )

        # B. 수수료 설정 (백테스트는 보수적으로 Taker Fee 적용)
        # config.py에 TAKER_FEE가 없으면 기존 COMMISSION 값(0.0005) 사용
        taker_fee = float(cfg.get('TAKER_FEE', 0.0005))
        
        # 수정된 FuturesComm 클래스 사용 (slippage_perc 파라미터 제거됨)
        cerebro.broker.addcommissioninfo(FuturesComm(
            commission=taker_fee,
            leverage=base_leverage
        ))
        # ------------------------------------------------------------------

        # 7. 전략 추가
        cerebro.addstrategy(HMM_Pro_Strategy_V5,
                            bull_id=bull_id, 
                            bear_id=bear_id, 
                            leverage=base_leverage,
                            max_positions=max_pos, 
                            rsi_buy_upper=float(cfg.get('RSI_BUY_UPPER', 65)),
                            rsi_buy_lower=float(cfg.get('RSI_BUY_LOWER', 30)),
                            rsi_sell_lower=float(cfg.get('RSI_SELL_LOWER', 35)),
                            rsi_sell_upper=float(cfg.get('RSI_SELL_UPPER', 70)),
                            stop_atr=float(cfg.get('STOP_LOSS_ATR', 2.0)),
                            trail_trigger=float(cfg.get('TRAIL_TRIGGER_ATR', 2.0)),
                            trail_dist=float(cfg.get('TRAIL_DIST_ATR', 2.0)),
                            cooldown_bars=2,
                            output_dir=output_dir 
                            )
        
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        
        log(f"🚀 [{symbol}] 시뮬레이션 실행 (기간: {test_days}일)...")
        results = cerebro.run()
        strat = results[0]
        current_cash = cerebro.broker.getvalue()
        
        if strat.trade_log:
            all_trade_logs = strat.trade_log
            
        mdd = strat.analyzers.drawdown.get_analysis()['max']['drawdown']
        csv_path = strat.csv_path

    except Exception as e:
        log(f"❌ 시뮬레이션 에러: {e}")
        return {"error": str(e)}

    # 8. 결과 정리
    total_roi = 0.0
    if initial_cash > 0:
        total_roi = (current_cash - initial_cash) / initial_cash * 100
    
    win_trades = [t for t in all_trade_logs if t['PnL'] > 0]
    win_rate = (len(win_trades) / len(all_trade_logs) * 100) if all_trade_logs else 0.0
    
    sorted_trades = sorted(all_trade_logs, key=lambda x: x['Exit_Date'])
    
    equity_curve = []
    equity_curve.append({"time": df_test.index[0].strftime('%Y-%m-%d'), "value": initial_cash})
    running_pnl = 0
    for t in sorted_trades:
        running_pnl += t['PnL']
        equity_curve.append({
            "time": t['Exit_Date'][:10],
            "value": initial_cash + running_pnl
        })
    equity_curve.append({"time": df_test.index[-1].strftime('%Y-%m-%d'), "value": current_cash})

    log(f"✅ [{symbol}] 완료: ROI {total_roi:+.2f}%, MDD {mdd:.2f}%")

    return {
        "symbol": symbol,
        "roi": round(total_roi, 2),
        "mdd": round(mdd, 2),
        "win_rate": round(win_rate, 2),
        "final_balance": round(current_cash, 2),
        "trade_count": len(all_trade_logs),
        "trades": sorted_trades,
        "equity_curve": equity_curve,
        "csv_path": csv_path,
        "params": cfg
    }

def run_batch_backtest(initial_cash=10000.0, train_days=365, test_days=30, update_data=False, log_func=None, dynamic_config=None):
    def log(msg):
        if log_func: log_func(msg)
        else: print(msg)
    
    cfg = dynamic_config if dynamic_config else CONFIG
    target_symbols = cfg.get('SYMBOLS', [])

    results = []
    log("🚀 전체 포트폴리오 백테스팅 (Simple Batch) 시작...")
    
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    batch_dir_name = f"Batch_{ts}"
    batch_dir_path = os.path.join(DATA_DIR, 'backtests', batch_dir_name)
    os.makedirs(batch_dir_path, exist_ok=True)
    
    for symbol in target_symbols:
        res = run_walk_forward(symbol, initial_cash, train_days, test_days, update_data, log_func, dynamic_config=cfg, output_dir=batch_dir_path)
        if "error" not in res: results.append(res)
    
    if not results: 
        if os.path.exists(batch_dir_path): shutil.rmtree(batch_dir_path)
        return {"error": "No results"}
    
    zip_path = ""
    try:
        df_list = []
        for res in results:
            if not res['equity_curve']: continue
            
            df = pd.DataFrame(res['equity_curve'])
            df['Date'] = pd.to_datetime(df['time'])
            df.set_index('Date', inplace=True)
            df.sort_index(inplace=True)
            
            df[res['symbol']] = df['value'] - initial_cash 
            df = df[[res['symbol']]] 
            df = df[~df.index.duplicated(keep='last')]
            df_list.append(df)
        
        portfolio_df = pd.concat(df_list, axis=1).sort_index()
        portfolio_df.fillna(method='ffill', inplace=True)
        portfolio_df.fillna(0, inplace=True)
        
        portfolio_df['Total_Equity'] = initial_cash + portfolio_df.sum(axis=1)
        
        summary_filename = f"Portfolio_Summary.csv"
        portfolio_df.reset_index(inplace=True)
        portfolio_df.to_csv(os.path.join(batch_dir_path, summary_filename), index=False)
        
        final_equity = portfolio_df['Total_Equity'].iloc[-1]
        pf_roi = (final_equity - initial_cash) / initial_cash * 100
        
        peak = portfolio_df['Total_Equity'].cummax()
        drawdown = (portfolio_df['Total_Equity'] - peak) / peak * 100
        pf_mdd = abs(drawdown.min())
        
        zip_base_name = os.path.join(DATA_DIR, 'backtests', batch_dir_name)
        zip_path_created = shutil.make_archive(zip_base_name, 'zip', root_dir=os.path.join(DATA_DIR, 'backtests'), base_dir=batch_dir_name)
        
        shutil.rmtree(batch_dir_path)
        zip_path = zip_path_created
        
    except Exception as e:
        log(f"⚠️ 포트폴리오 병합 및 압축 중 오류: {e}")
        pf_roi = 0
        pf_mdd = 0

    avg_roi = sum(r['roi'] for r in results) / len(results)
    
    log(f"🏆 [종합] Portfolio ROI: {pf_roi:+.2f}%, Portfolio MDD: {pf_mdd:.2f}%")
    
    return {
        "type": "batch", 
        "avg_roi": round(avg_roi, 2), 
        "portfolio_roi": round(pf_roi, 2), 
        "avg_mdd": round(pf_mdd, 2),
        "details": results,
        "csv_path": zip_path 
    }

# --------------------------------------------------------- 
# [NEW] 리얼 포트폴리오 실행 함수 (ZeroDivisionError 방지 패치)
# --------------------------------------------------------- 
def run_real_portfolio_backtest(initial_cash=10000.0, train_days=365, test_days=30, update_data=False, log_func=None, dynamic_config=None):
    def log(msg):
        if log_func: log_func(msg)
        else: print(msg)

    cfg = dynamic_config if dynamic_config else CONFIG
    target_symbols = cfg.get('SYMBOLS', [])
    
    log(f"🚀 Real Portfolio Backtest 시작 (Symbols: {len(target_symbols)}개, 자금: ${initial_cash})")

    if update_data:
        log("📥 최신 데이터 업데이트 중...")
        try:
            loader = MultiSymbolLoader()
            required_days = train_days + test_days + 100
            fetch_days = max(cfg.get('FETCH_DAYS', 1500), required_days)
            for sym in target_symbols:
                df, path = loader.fetch_ohlcv(sym, days=fetch_days)
                if not df.empty:
                    df = loader.add_features(df)
                    df.to_csv(path)
        except Exception as e:
            log(f"❌ 데이터 업데이트 실패: {e}")
            return {"error": str(e)}

    # [수정] 브로커 설정 고도화 (슬리피지 및 Taker Fee 적용)
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(initial_cash)
    
    # 1. 안전한 레버리지 설정
    lev = float(cfg.get('LEVERAGE', 2.0))
    if lev <= 0: lev = 1.0

    # 2. 슬리피지 설정 (가격 왜곡 적용)
    cerebro.broker.set_slippage_perc(
        perc=float(cfg.get('SLIPPAGE_PCT', 0.0002)),
        slip_open=True,
        slip_match=True,
        slip_out=False
    )
    
    # 3. 수수료 설정 (보수적 Taker Fee)
    taker_fee = float(cfg.get('TAKER_FEE', 0.0005))
    
    cerebro.broker.addcommissioninfo(FuturesComm(
        commission=taker_fee,
        leverage=lev
    ))
    
    regime_map_all = {} 
    brain = Brain()
    valid_data_count = 0
    
    for sym in target_symbols:
        df_full, _ = brain.load_data(sym)
        if df_full is None or len(df_full) < (train_days + test_days): continue

        end_date = df_full.index[-1]
        test_start = end_date - timedelta(days=test_days)
        train_end = test_start - timedelta(seconds=1)
        train_start = train_end - timedelta(days=train_days)
        
        if train_start < df_full.index[0]: train_start = df_full.index[0]

        df_train = df_full.loc[train_start:train_end].copy()
        if len(df_train) < 100: continue
        
        try:
            model, df_train_res = brain.train_model(df_train, sym)
            r_map, _ = brain.identify_regimes(df_train_res)
            
            bull_idx = [k for k, v in r_map.items() if 'Bull' in v][0]
            bear_idx = [k for k, v in r_map.items() if 'Bear' in v][0]
            regime_map_all[sym] = {'bull': bull_idx, 'bear': bear_idx}

            df_test = df_full.loc[test_start:end_date].copy()
            if len(df_test) < 10: continue
            
            test_features = df_test[['Log_Returns_Scaled', 'Range_Vol_Scaled', 'RSI_14_Scaled', 'OBV_Scaled']].values
            df_test['Regime'] = model.predict(test_features)
            
            data_feed = HMMData(dataname=df_test, name=sym)
            cerebro.adddata(data_feed)
            valid_data_count += 1
            
        except Exception as e:
            log(f"⚠️ {sym} 처리 중 오류: {e}")
            continue

    if valid_data_count == 0:
        return {"error": "No valid data found"}

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    real_dir_name = f"RealPF_{ts}"
    real_dir_path = os.path.join(DATA_DIR, 'backtests', real_dir_name)
    os.makedirs(real_dir_path, exist_ok=True)

    cerebro.addstrategy(RealPortfolioStrategy, 
                        regime_map=regime_map_all,
                        config=cfg,
                        output_dir=real_dir_path)

    log("🏃 시뮬레이션 진행 중...")
    results = cerebro.run()
    strat = results[0]
    
    final_value = cerebro.broker.getvalue()
    
    # [FIX] 안전한 ROI 계산 (Initial Cash 0 방지)
    roi = 0.0
    if initial_cash > 0:
        roi = (final_value - initial_cash) / initial_cash * 100
    
    zip_path = ""
    trade_count = 0
    equity_curve = []
    
    try:
        trade_count = len(strat.trade_records)
        
        if strat.daily_stats:
            df_equity = pd.DataFrame(strat.daily_stats)[['Date', 'Total_Equity']]
            df_equity.rename(columns={'Date': 'time', 'Total_Equity': 'value'}, inplace=True)
            df_equity.to_csv(os.path.join(real_dir_path, 'Portfolio_Curve.csv'), index=False)
            equity_curve = df_equity.to_dict(orient='records')

        zip_base_name = os.path.join(DATA_DIR, 'backtests', real_dir_name)
        zip_path_created = shutil.make_archive(zip_base_name, 'zip', root_dir=os.path.join(DATA_DIR, 'backtests'), base_dir=real_dir_name)
        
        shutil.rmtree(real_dir_path)
        zip_path = zip_path_created
    except Exception as e:
        log(f"⚠️ 결과 저장 중 오류: {e}")

    mdd = 0.0
    if equity_curve:
        df = pd.DataFrame(equity_curve)
        peak = df['value'].cummax()
        # [FIX] Peak가 0일 경우 방지
        peak = peak.replace(0, 1) 
        dd = (df['value'] - peak) / peak * 100
        mdd = abs(dd.min())

    log(f"🏁 Real Portfolio 완료. ROI: {roi:+.2f}%, MDD: {mdd:.2f}%")

    return {
        "type": "real_pf",
        "portfolio_roi": round(roi, 2),
        "avg_mdd": round(mdd, 2),
        "avg_roi": round(roi, 2),
        "trade_count": trade_count,
        "final_balance": round(final_value, 2),
        "csv_path": zip_path,
        "equity_curve": equity_curve
    }