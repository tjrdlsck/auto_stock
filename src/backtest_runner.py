import sys
import os
import shutil
import backtrader as bt
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict # [NEW] PnL 추적용
import joblib 

# 프로젝트 루트 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import CONFIG, DATA_DIR
from alpha_layer.brain import Brain
from data_layer.data_handler import MultiSymbolLoader

# --------------------------------------------------------- 
# 1. 데이터 피드 및 공통 클래스 (기존 유지)
# --------------------------------------------------------- 
# [수정] CSV의 지표 값을 그대로 사용하는 데이터 피드
class HMMData(bt.feeds.PandasData):
    # CSV에 있는 컬럼명(features.py 결과)과 매핑할 라인 정의
    lines = ('rsi', 'ema', 'atr', 'regime',)
    
    params = (
        ('datetime', None), # 인덱스 사용
        ('open', 'open'),
        ('high', 'high'),
        ('low', 'low'),
        ('close', 'close'),
        ('volume', 'volume'),
        ('openinterest', -1),
        
        # [NEW] 지표 매핑 (features.py의 컬럼명과 일치해야 함)
        ('rsi', 'RSI_14'),     # Raw RSI (30/70 기준)
        ('ema', 'EMA_20'),     # EMA
        ('atr', 'ATRr_14'),    # Running ATR
        ('regime', 'Regime'),  # 외부에서 예측하여 주입할 국면 데이터
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
        ('regime_map', {}), 
        ('leverage', 2.0),
        ('max_positions', 3),
        ('rsi_buy_upper', 65), ('rsi_buy_lower', 30),
        ('rsi_sell_lower', 35), ('rsi_sell_upper', 70),
        ('stop_atr', 2.0), ('trail_trigger', 2.0), ('trail_dist', 2.0),
        ('cooldown_bars', 2),
        ('output_dir', None),
        ('funding_rate', 0.0001),
        ('full_df', None),
        ('train_days', 365),
        ('test_days', 30),
        ('brain', None),
        ('symbol_name', ''),
        # [수정] 진행률 보고를 위한 파라미터 추가
        ('progress_callback', None),
        ('total_steps', 0)
    )

    def __init__(self):
        self.atr = self.data.atr
        self.rsi = self.data.rsi
        self.ema = self.data.ema
        self.order = None
        self.entry_price = 0.0
        self.stop_price = 0.0
        self.cooldown = 0
        self.entry_metadata = {} 
        self.exit_reason = "Manual/End" 
        self.trade_log = []   
        self.daily_stats = []
        self.model = None
        self.current_regime = None
        self.regime_map = {}
        self.last_retrain_date = None
        self._initial_train()

    def _initial_train(self):
        current_dt = self.data.datetime.date(0)
        self._retrain_model(current_dt)

    def _retrain_model(self, current_dt):
        if self.p.full_df is None or self.p.brain is None: return
        try:
            train_start = current_dt - timedelta(days=self.p.train_days)
            mask = (self.p.full_df.index >= pd.Timestamp(train_start)) & (self.p.full_df.index < pd.Timestamp(current_dt))
            train_data = self.p.full_df.loc[mask].copy()
            if len(train_data) < 100: return
            self.model, _ = self.p.brain.train_model(train_data, self.p.symbol_name)
            r_map, _ = self.p.brain.identify_regimes(train_data)
            bull_idx = [k for k, v in r_map.items() if 'Bull' in v][0]
            bear_idx = [k for k, v in r_map.items() if 'Bear' in v][0]
            self.regime_map = {'bull': bull_idx, 'bear': bear_idx}
            self.last_retrain_date = current_dt
        except: pass

    def notify_order(self, order):
        if order.status in [order.Completed]:
            executed_price = order.executed.price
            executed_size = abs(order.executed.size)
            if order.isbuy():
                if self.position.size > 0: 
                    self.entry_price = executed_price
                    self.stop_price = self.entry_price - (self.atr[0] * self.p.stop_atr)
                    self.entry_metadata = {'size': executed_size, 'entry_time': self.data.datetime.datetime(0), 'regime': 'Bull'}
            elif order.issell():
                if self.position.size < 0:
                    self.entry_price = executed_price
                    self.stop_price = self.entry_price + (self.atr[0] * self.p.stop_atr)
                    self.entry_metadata = {'size': executed_size, 'entry_time': self.data.datetime.datetime(0), 'regime': 'Bear'}
            self.order = None
            self.cooldown = 0
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed: return
        real_size = self.entry_metadata.get('size', 0.0)
        if real_size == 0 and trade.history: real_size = abs(trade.history[0].event.size)
        entry_time = self.entry_metadata.get('entry_time', bt.num2date(trade.dtopen))
        pnl_net = trade.pnlcomm
        safe_lev = self.p.leverage if self.p.leverage > 0 else 1.0
        entry_val = (trade.price * real_size) / safe_lev
        roi = (pnl_net / entry_val) * 100 if entry_val != 0 else 0.0
        
        exit_price = trade.price
        if real_size > 0:
            exit_price = trade.price + (trade.pnl / real_size) if trade.long else trade.price - (trade.pnl / real_size)

        self.trade_log.append({
            'Symbol': self.data._name, 'Type': 'Long' if trade.long else 'Short',
            'Entry_Time': entry_time, 'Exit_Time': bt.num2date(trade.dtclose),
            'Duration_Hours': (bt.num2date(trade.dtclose) - entry_time).total_seconds()/3600,
            'Entry_Price': trade.price, 'Exit_Price': exit_price,
            'Size': real_size, 'PnL': pnl_net, 'Fee': trade.commission,
            'ROI': roi, 'Entry_Regime': self.entry_metadata.get('regime', 'Unknown'),
            'Exit_Reason': self.exit_reason
        })
        self.exit_reason = "Manual/End"

    def next(self):
        # [수정] 진행률 보고 로직 (전체의 5% ~ 100% 구간 담당)
        # 준비 단계(5%) 이후 나머지 95%를 채웁니다.
        if self.p.progress_callback and len(self) % 24 == 0:
            current_step = len(self)
            total = self.p.total_steps
            # 공식 변경: 5%에서 시작 + (진행률 * 95%)
            progress = 5 + int((current_step / total) * 95)
            # 100%를 넘지 않도록 안전장치
            if progress > 99: progress = 99
            self.p.progress_callback(progress)

        # ... (이하 기존 로직 동일) ...
        current_dt = self.data.datetime.date(0)
        current_dt_full = self.data.datetime.datetime(0)

        if self.last_retrain_date is None: self._retrain_model(current_dt)
        elif (current_dt - self.last_retrain_date).days >= self.p.test_days: self._retrain_model(current_dt)

        if self.model:
            try:
                ts = pd.Timestamp(current_dt_full)
                if ts in self.p.full_df.index:
                    row = self.p.full_df.loc[ts]
                    X_curr = [[row['Log_Returns_Scaled'], row['Range_Vol_Scaled'], row['RSI_14_Scaled'], row['OBV_Scaled']]]
                    self.current_regime = int(self.model.predict(X_curr)[0])
            except: pass

        self.daily_stats.append({
            'Date': current_dt_full, 'Close': self.data.close[0],
            'Regime': self.current_regime if self.current_regime is not None else -1,
            'Position_Size': self.position.size, 'Portfolio_Value': self.broker.getvalue()
        })
        
        if current_dt_full.minute == 0 and current_dt_full.hour in [0, 8, 16]:
            if self.position.size != 0:
                cost = abs(self.position.size) * self.data.close[0] * self.p.funding_rate
                self.broker.add_cash(-cost)
                self.trade_log.append({
                    'Symbol': self.data._name, 'Type': 'Funding',
                    'Entry_Time': current_dt_full, 'Exit_Time': current_dt_full,
                    'Duration_Hours': 0, 'Entry_Price': 0, 'Exit_Price': 0,
                    'Size': 0, 'PnL': -cost, 'Fee': 0, 'ROI': 0,
                    'Entry_Regime': '-', 'Exit_Reason': 'Funding Fee'
                })

        if self.order: return
        if self.current_regime is None or not self.regime_map: return

        is_bull = (self.current_regime == self.regime_map.get('bull'))
        is_bear = (self.current_regime == self.regime_map.get('bear'))
        close = self.data.close[0]
        atr = self.atr[0]
        rsi = self.rsi[0]
        ema = self.ema[0]
        if self.cooldown > 0: self.cooldown -= 1

        if self.position.size != 0:
            should_close = False
            reason = ""
            if self.position.size > 0:
                if self.data.low[0] < self.stop_price: should_close = True; reason = "StopLoss"
                elif close > self.entry_price + (atr * self.p.trail_trigger):
                    new_stop = close - (atr * self.p.trail_dist)
                    if new_stop > self.stop_price: self.stop_price = new_stop
                elif is_bear: should_close = True; reason = "RegimeChange(Bear)"
            elif self.position.size < 0:
                if self.data.high[0] > self.stop_price: should_close = True; reason = "StopLoss"
                elif close < self.entry_price - (atr * self.p.trail_trigger):
                    new_stop = close + (atr * self.p.trail_dist)
                    if new_stop < self.stop_price: self.stop_price = new_stop
                elif is_bull: should_close = True; reason = "RegimeChange(Bull)"
            
            if should_close:
                self.exit_reason = reason; self.close(); self.cooldown = self.p.cooldown_bars; return

        elif self.position.size == 0 and self.cooldown == 0:
            cash = self.broker.get_cash()
            if cash <= 0: return
            alloc = self.broker.getvalue() / self.p.max_positions
            qty = (alloc * 0.95 * self.p.leverage) / close if close > 0 else 0
            if qty < 0.000001: return

            if is_bull and close > ema and self.p.rsi_buy_lower < rsi < self.p.rsi_buy_upper:
                self.order = self.buy(size=qty)
            elif is_bear and close < ema and self.p.rsi_sell_lower < rsi < self.p.rsi_sell_upper:
                self.order = self.sell(size=qty)

    def stop(self):
        if self.position.size != 0:
            size = self.position.size; curr = self.data.close[0]; entry = self.position.price
            raw_pnl = (curr - entry) * abs(size) if size > 0 else (entry - curr) * abs(size)
            fee = abs(size) * curr * 0.0005
            margin = (entry * abs(size)) / self.p.leverage
            roi = ((raw_pnl - fee) / margin) * 100 if margin > 0 else 0
            self.trade_log.append({
                'Symbol': self.data._name, 'Type': 'Long' if size > 0 else 'Short',
                'Entry_Time': self.entry_metadata.get('entry_time', 'Unknown'),
                'Exit_Time': self.data.datetime.datetime(0), 'Duration_Hours': 0,
                'Entry_Price': entry, 'Exit_Price': curr, 'Size': abs(size),
                'PnL': raw_pnl - fee, 'Fee': fee, 'ROI': roi,
                'Entry_Regime': self.entry_metadata.get('regime', 'Unknown'), 'Exit_Reason': "Test End (Open)"
            })
        
        save_dir = self.p.output_dir
        if not save_dir: return
        if self.trade_log:
            df = pd.DataFrame(self.trade_log)
            if 'Entry_Time' in df.columns: df.sort_values(by='Entry_Time', inplace=True)
            df.to_csv(os.path.join(save_dir, 'Master_Trade_Log.csv'), index=False)
        if self.daily_stats:
            df = pd.DataFrame(self.daily_stats)
            df.drop_duplicates(subset=['Date'], keep='last', inplace=True)
            df.to_csv(os.path.join(save_dir, 'Daily_Portfolio_Stats.csv'), index=False)


# --------------------------------------------------------- 
# [NEW] 리얼 포트폴리오 전략 (Size 0 버그 수정판)
# --------------------------------------------------------- 
class RealPortfolioStrategy(bt.Strategy):
    params = (
        ('regime_map', {}), ('max_positions', 3), ('leverage', 2.0),
        ('config', {}), ('output_dir', None),
        ('full_dfs', {}), ('train_days', 365), ('test_days', 30), ('brain', None),
        # [수정] 진행률 파라미터
        ('progress_callback', None), ('total_steps', 0)
    )

    def __init__(self):
        self.inds = {}; self.orders = {}; self.cooldowns = {}; self.stops = {}
        self.entry_metadata = {}; self.exit_reasons = {}
        self.trade_records = []; self.daily_stats = []
        self.pnl_history = defaultdict(list); self.realized_pnl = defaultdict(float)
        self.models = {}; self.regime_maps = {}; self.last_retrain = {}

        for d in self.datas:
            name = d._name
            self.orders[name] = None; self.cooldowns[name] = 0; self.stops[name] = 0.0
            self.entry_metadata[name] = {}; self.exit_reasons[name] = "Manual/End"
            self.last_retrain[name] = None
            cfg = self.p.config
            self.inds[name] = {
                'atr': bt.indicators.ATR(d, period=int(cfg.get('ATR_PERIOD', 14))),
                'rsi': bt.indicators.RSI(d, period=int(cfg.get('RSI_PERIOD', 14))),
                'ema': bt.indicators.EMA(d.close, period=int(cfg.get('EMA_PERIOD', 20))),
                'curr_regime': None
            }

    def _retrain_symbol(self, symbol, current_dt):
        if not self.p.brain or symbol not in self.p.full_dfs: return
        try:
            full_df = self.p.full_dfs[symbol]
            train_start = current_dt - timedelta(days=self.p.train_days)
            mask = (full_df.index >= pd.Timestamp(train_start)) & (full_df.index < pd.Timestamp(current_dt))
            train_data = full_df.loc[mask].copy()
            if len(train_data) < 100: return
            model, _ = self.p.brain.train_model(train_data, symbol)
            r_map, _ = self.p.brain.identify_regimes(train_data)
            self.models[symbol] = model
            self.regime_maps[symbol] = {'bull': [k for k,v in r_map.items() if 'Bull' in v][0], 
                                        'bear': [k for k,v in r_map.items() if 'Bear' in v][0]}
            self.last_retrain[symbol] = current_dt
        except: pass

    def notify_order(self, order):
        name = order.data._name
        if order.status in [order.Completed]:
            if (order.isbuy() and self.getposition(order.data).size > 0) or \
               (order.issell() and self.getposition(order.data).size < 0):
                r_map = self.regime_maps.get(name)
                curr = self.inds[name]['curr_regime']
                reg_str = 'Bull' if r_map and curr == r_map['bull'] else 'Bear' if r_map and curr == r_map['bear'] else 'Unknown'
                self.entry_metadata[name] = {'entry_price': order.executed.price, 'entry_time': order.data.datetime.datetime(0), 'size': abs(order.executed.size), 'regime': reg_str}
                stop_atr = float(self.p.config.get('STOP_LOSS_ATR', 2.0))
                atr = self.inds[name]['atr'][0]
                if order.isbuy(): self.stops[name] = order.executed.price - (atr * stop_atr)
                else: self.stops[name] = order.executed.price + (atr * stop_atr)
            self.orders[name] = None; self.cooldowns[name] = 0
        elif order.status in [order.Canceled, order.Margin, order.Rejected]: self.orders[name] = None

    def notify_trade(self, trade):
        if not trade.isclosed: return
        name = trade.data._name
        pnl = trade.pnlcomm; self.realized_pnl[name] += pnl
        meta = self.entry_metadata.get(name, {})
        entry_time = meta.get('entry_time', bt.num2date(trade.dtopen))
        size = meta.get('size', abs(trade.history[0].event.size) if trade.history else 0)
        exit_time = bt.num2date(trade.dtclose)
        dur = (exit_time - entry_time).total_seconds() / 3600
        lev = self.p.leverage if self.p.leverage > 0 else 1.0
        roi = (pnl / ((trade.price * size)/lev)) * 100 if size > 0 else 0
        exit_p = trade.price + (trade.pnl/size) if trade.long else trade.price - (trade.pnl/size)
        self.trade_records.append({
            'Symbol': name, 'Side': 'Long' if trade.long else 'Short',
            'Entry_Time': entry_time, 'Exit_Time': exit_time, 'Duration_Hours': dur,
            'Entry_Price': trade.price, 'Exit_Price': exit_p, 'Size': size, 'PnL_Net': pnl,
            'Fee': trade.commission, 'ROI_Pct': roi, 'Entry_Regime': meta.get('regime', 'Unknown'),
            'Exit_Reason': self.exit_reasons.get(name, 'Unknown')
        })

    def next(self):
        # [수정] 진행률 보고 (5% ~ 99%)
        if self.p.progress_callback and len(self) % 24 == 0:
            current_step = len(self)
            total = self.p.total_steps
            progress = 5 + int((current_step / total) * 95)
            if progress > 99: progress = 99
            self.p.progress_callback(progress)

        # ... (이하 기존 로직 동일) ...
        dt = self.datas[0].datetime.datetime(0)
        dt_date = self.datas[0].datetime.date(0)
        open_pos = sum(1 for d in self.datas if self.getposition(d).size != 0)
        
        self.daily_stats.append({'Date': dt, 'Total_Equity': self.broker.getvalue(), 'Cash': self.broker.getcash(), 'Open_Positions': open_pos, 'Leverage': self.p.leverage})
        for d in self.datas:
            name = d._name; pos = self.getposition(d)
            unrealized = pos.size * (d.close[0] - pos.price) if pos.size != 0 else 0
            self.pnl_history[name].append({'Date': dt, 'Portfolio_Value': self.realized_pnl[name] + unrealized})

        cfg = self.p.config; max_pos = int(cfg.get('MAX_OPEN_POSITIONS', 3))
        
        for d in self.datas:
            name = d._name
            if self.last_retrain[name] is None or (dt_date - self.last_retrain[name]).days >= self.p.test_days:
                self._retrain_symbol(name, dt_date)
            
            curr_regime = None
            model = self.models.get(name)
            if model and name in self.p.full_dfs:
                try:
                    ts = pd.Timestamp(dt)
                    if ts in self.p.full_dfs[name].index:
                        row = self.p.full_dfs[name].loc[ts]
                        curr_regime = int(model.predict([[row['Log_Returns_Scaled'], row['Range_Vol_Scaled'], row['RSI_14_Scaled'], row['OBV_Scaled']]])[0])
                except: pass
            self.inds[name]['curr_regime'] = curr_regime
            r_map = self.regime_maps.get(name)
            if not r_map or curr_regime is None: continue

            pos = self.getposition(d); close = d.close[0]
            atr = self.inds[name]['atr'][0]; rsi = self.inds[name]['rsi'][0]; ema = self.inds[name]['ema'][0]
            
            if self.cooldowns[name] > 0: self.cooldowns[name] -= 1

            if pos.size != 0:
                stop = self.stops[name]; should_close = False; reason = ""
                trig = float(cfg.get('TRAIL_TRIGGER_ATR', 2.0)); dist = float(cfg.get('TRAIL_DIST_ATR', 2.0))
                if pos.size > 0:
                    if d.low[0] < stop: should_close = True; reason = "StopLoss"
                    elif close > self.entry_metadata[name].get('entry_price', 0) + (atr * trig):
                        if (close - atr*dist) > stop: self.stops[name] = close - atr*dist
                    elif curr_regime == r_map['bear']: should_close = True; reason = "Regime(Bear)"
                else:
                    if d.high[0] > stop: should_close = True; reason = "StopLoss"
                    elif close < self.entry_metadata[name].get('entry_price', 0) - (atr * trig):
                        if (close + atr*dist) < stop: self.stops[name] = close + atr*dist
                    elif curr_regime == r_map['bull']: should_close = True; reason = "Regime(Bull)"
                if should_close: self.exit_reasons[name] = reason; self.close(data=d); self.cooldowns[name] = 2

            elif self.orders[name] is None and self.cooldowns[name] == 0:
                if open_pos >= max_pos: continue
                alloc = self.broker.getvalue() / max_pos
                qty = (alloc * 0.95 * self.p.leverage) / close if close > 0 else 0
                if qty <= 0: continue
                if curr_regime == r_map['bull'] and close > ema and float(cfg.get('RSI_BUY_LOWER',30)) < rsi < float(cfg.get('RSI_BUY_UPPER',65)):
                    self.orders[name] = self.buy(data=d, size=qty); open_pos += 1
                elif curr_regime == r_map['bear'] and close < ema and float(cfg.get('RSI_SELL_LOWER',35)) < rsi < float(cfg.get('RSI_SELL_UPPER',70)):
                    self.orders[name] = self.sell(data=d, size=qty); open_pos += 1

    def stop(self):
        for d in self.datas:
            name = d._name; pos = self.getposition(d)
            if pos.size != 0:
                size = pos.size; curr = d.close[0]; entry = pos.price
                raw = (curr - entry) * abs(size) if size > 0 else (entry - curr) * abs(size)
                fee = abs(size) * curr * 0.0005; margin = (entry * abs(size)) / self.p.leverage
                roi = ((raw - fee) / margin) * 100 if margin > 0 else 0
                self.trade_records.append({
                    'Symbol': name, 'Side': 'Long' if size > 0 else 'Short',
                    'Entry_Time': self.entry_metadata.get(name, {}).get('entry_time', 'Unknown'),
                    'Exit_Time': d.datetime.datetime(0), 'Duration_Hours': 0,
                    'Entry_Price': entry, 'Exit_Price': curr, 'Size': abs(size),
                    'PnL_Net': raw - fee, 'Fee': fee, 'ROI_Pct': roi,
                    'Entry_Regime': self.entry_metadata.get(name, {}).get('regime', 'Unknown'),
                    'Exit_Reason': "Test End (Open)"
                })
        save_dir = self.p.output_dir
        if save_dir:
            if self.trade_records:
                pd.DataFrame(self.trade_records).sort_values('Entry_Time').to_csv(os.path.join(save_dir, 'Master_Trade_Log.csv'), index=False)
            if self.daily_stats:
                pd.DataFrame(self.daily_stats).drop_duplicates('Date', keep='last').to_csv(os.path.join(save_dir, 'Daily_Portfolio_Stats.csv'), index=False)


# --------------------------------------------------------- 
# [EXISTING] 기존 실행 함수들 (유지)
# --------------------------------------------------------- 
# [수정] 날짜 지정 기능이 포함된 Walk-Forward 엔진
def run_walk_forward(symbol, initial_cash=10000.0, train_days=1500, test_days=30, 
                     start_date=None, end_date=None, 
                     update_data=False, log_func=None, dynamic_config=None, output_dir=None,
                     progress_callback=None): 
    def log(msg):
        if log_func: log_func(msg)
        else: print(msg)

    if test_days <= 0:
        return {"error": f"재학습 주기(test_days)는 0일 수 없습니다. 최소 1일 이상 설정해주세요."}
    
    cfg = dynamic_config if dynamic_config else CONFIG
    
    # 1. 데이터 업데이트
    if update_data:
        log(f"📥 [{symbol}] 데이터 업데이트 확인 중...")
        try:
            loader = MultiSymbolLoader()
            # 넉넉하게 fetch
            fetch_days = max(cfg.get('FETCH_DAYS', 2500), train_days + test_days + 365)
            df, path = loader.fetch_ohlcv(symbol, days=fetch_days)
            if not df.empty:
                df = loader.add_features(df)
                df.to_csv(path)
            else:
                return {"error": "Data fetch failed"}
        except Exception as e:
            return {"error": str(e)}

    # 2. 데이터 로드 (Brain 이용)
    brain = Brain()
    df_full, _ = brain.load_data(symbol)
    if df_full is None or df_full.empty:
        return {"error": "Data load failed"}
    
    df_full.sort_index(inplace=True)
    
    # 3. 기간 설정
    data_min_date = df_full.index[0]
    data_max_date = df_full.index[-1]
    
    # 최소 학습 기간 확보 확인
    min_required = timedelta(days=train_days)
    if (data_max_date - data_min_date) < min_required:
         return {"error": f"데이터 부족. 보유: {(data_max_date - data_min_date).days}일 vs 필요: {train_days}일"}

    # 테스트 시작일 설정 (최초 학습 기간 이후)
    req_start_date = data_min_date + min_required
    if start_date:
        try:
            parsed = pd.Timestamp(start_date)
            if parsed > req_start_date: req_start_date = parsed
        except: pass

    req_end_date = data_max_date
    if end_date:
        try:
            parsed = pd.Timestamp(end_date)
            if parsed < data_max_date: req_end_date = parsed
        except: pass

    if req_start_date >= req_end_date:
        return {"error": "유효하지 않은 기간 설정 (Start >= End)"}

    log(f"🚀 [{symbol}] Dynamic Walk-Forward 시작 (In-Strategy Training)")
    
    # 4. 테스트 기간 데이터 슬라이싱
    mask_test = (df_full.index >= req_start_date) & (df_full.index <= req_end_date)
    test_data = df_full.loc[mask_test].copy()
    
    if len(test_data) == 0: return {"error": "No data in selected range"}

    # 5. Cerebro 설정
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(initial_cash)
    
    taker_fee = float(cfg.get('TAKER_FEE', 0.0005))
    cerebro.broker.set_slippage_perc(float(cfg.get('SLIPPAGE_PCT', 0.0002)))
    cerebro.broker.addcommissioninfo(FuturesComm(commission=taker_fee, leverage=float(cfg.get('LEVERAGE', 2.0))))

    data_feed = HMMData(dataname=test_data, name=symbol)
    cerebro.adddata(data_feed)

    # 6. 진행률 콜백 래퍼
    def strategy_progress_wrapper(p):
        if progress_callback:
            # 전략 내부에서는 p가 5~99 사이 값으로 계산되어 옴
            progress_callback(p, 100, f"Running... {p}%")

    # 7. 전략 추가 (전체 데이터와 Brain 주입)
    cerebro.addstrategy(HMM_Pro_Strategy_V5,
                        leverage=float(cfg.get('LEVERAGE', 2.0)),
                        max_positions=int(cfg.get('MAX_OPEN_POSITIONS', 3)),
                        rsi_buy_upper=float(cfg.get('RSI_BUY_UPPER', 65)),
                        rsi_buy_lower=float(cfg.get('RSI_BUY_LOWER', 30)),
                        rsi_sell_lower=float(cfg.get('RSI_SELL_LOWER', 35)),
                        rsi_sell_upper=float(cfg.get('RSI_SELL_UPPER', 70)),
                        stop_atr=float(cfg.get('STOP_LOSS_ATR', 2.0)),
                        trail_trigger=float(cfg.get('TRAIL_TRIGGER_ATR', 2.0)),
                        trail_dist=float(cfg.get('TRAIL_DIST_ATR', 2.0)),
                        output_dir=output_dir,
                        # [In-Strategy Params]
                        full_df=df_full,
                        train_days=train_days,
                        test_days=test_days,
                        brain=brain,
                        symbol_name=symbol,
                        progress_callback=strategy_progress_wrapper,
                        total_steps=len(test_data)
                        )

    # 8. 실행
    if progress_callback: progress_callback(5, 100, "Initializing Strategy...") # 5% 시작
    results = cerebro.run()
    if progress_callback: progress_callback(100, 100, "Done")
    
    # 9. 결과 정리
    strat = results[0]
    final_val = cerebro.broker.getvalue()
    all_trades = strat.trade_log
    
    equity_curve = []
    for stat in strat.daily_stats:
        equity_curve.append({
            "time": stat['Date'].strftime('%Y-%m-%d'),
            "value": stat['Portfolio_Value']
        })

    total_roi = (final_val - initial_cash) / initial_cash * 100
    win_trades = [t for t in all_trades if t.get('PnL', 0) > 0]
    win_rate = (len(win_trades) / len(all_trades) * 100) if all_trades else 0.0
    
    # 10. CSV 저장 (Equity 계산 포함)
    df_trades = pd.DataFrame(all_trades)
    mdd = 0.0
    csv_save_path = ""
    
    if output_dir:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_save_path = os.path.join(output_dir, f"WFA_{symbol.replace('/','')}_{ts}.csv")
        
        if not df_trades.empty:
            df_trades.sort_values(by='Entry_Time', inplace=True)
            
            # PnL 기반 Equity 재계산 (검증용)
            if 'PnL' in df_trades.columns:
                df_trades['cum_pnl'] = df_trades['PnL'].cumsum()
                df_trades['equity'] = initial_cash + df_trades['cum_pnl']
                
                peak = df_trades['equity'].cummax()
                dd = (df_trades['equity'] - peak) / peak * 100
                mdd = abs(dd.min())
            
            # 그래프 로더 호환성
            if 'Entry_Time' in df_trades.columns:
                df_trades['time'] = df_trades['Entry_Time']
            
            df_trades.to_csv(csv_save_path, index=False)
            
    log(f"🏁 [{symbol}] 완료. Cash: ${final_val:.2f}, ROI: {total_roi:.2f}%")

    return {
        "symbol": symbol, "roi": round(total_roi, 2), "mdd": round(mdd, 2),
        "win_rate": round(win_rate, 2), "final_balance": round(final_val, 2),
        "trade_count": len(all_trades), "trades": all_trades,
        "equity_curve": equity_curve, "csv_path": csv_save_path, "params": cfg
    }

def run_batch_backtest(initial_cash=10000.0, train_days=365, test_days=30, 
                       start_date=None, end_date=None, 
                       update_data=False, log_func=None, dynamic_config=None,
                       progress_callback=None): 
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
    
    total_syms = len(target_symbols)

    for i, symbol in enumerate(target_symbols):
        if progress_callback:
            progress_callback(i, total_syms, f"Batch Processing: {symbol} ({i+1}/{total_syms})")

        res = run_walk_forward(symbol, initial_cash, train_days, test_days, 
                               start_date=start_date, end_date=end_date, 
                               update_data=update_data, log_func=log_func, 
                               dynamic_config=cfg, output_dir=batch_dir_path)
        if "error" not in res: results.append(res)
    
    if progress_callback:
        progress_callback(total_syms, total_syms, "Batch Complete")

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
        
        if df_list:
            portfolio_df = pd.concat(df_list, axis=1).sort_index()
            portfolio_df.fillna(method='ffill', inplace=True)
            portfolio_df.fillna(0, inplace=True)
            
            portfolio_df['Total_Equity'] = initial_cash + portfolio_df.sum(axis=1)
            
            summary_filename = f"Portfolio_Summary.csv"
            portfolio_df.reset_index(inplace=True)
            
            # [수정] 컬럼명 표준화 (Date -> time)
            portfolio_df.rename(columns={'Date': 'time'}, inplace=True)
            
            portfolio_df.to_csv(os.path.join(batch_dir_path, summary_filename), index=False)
            
            final_equity = portfolio_df['Total_Equity'].iloc[-1]
            pf_roi = (final_equity - initial_cash) / initial_cash * 100
            
            peak = portfolio_df['Total_Equity'].cummax()
            drawdown = (portfolio_df['Total_Equity'] - peak) / peak * 100
            pf_mdd = abs(drawdown.min())
        else:
            pf_roi = 0
            pf_mdd = 0

        zip_base_name = os.path.join(DATA_DIR, 'backtests', batch_dir_name)
        zip_path_created = shutil.make_archive(zip_base_name, 'zip', root_dir=os.path.join(DATA_DIR, 'backtests'), base_dir=batch_dir_name)
        
        shutil.rmtree(batch_dir_path)
        zip_path = zip_path_created
        
    except Exception as e:
        log(f"⚠️ 포트폴리오 병합 및 압축 중 오류: {e}")
        pf_roi = 0
        pf_mdd = 0
        zip_path = ""

    avg_roi = sum(r['roi'] for r in results) / len(results) if results else 0
    
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
# [수정] 리얼 포트폴리오 모드에도 Walk-Forward(재학습) 엔진 탑재
def run_real_portfolio_backtest(initial_cash=10000.0, train_days=1500, test_days=30, 
                                start_date=None, end_date=None, 
                                update_data=False, log_func=None, dynamic_config=None,
                                progress_callback=None):
    def log(msg):
        if log_func: log_func(msg)
        else: print(msg)

    cfg = dynamic_config if dynamic_config else CONFIG
    target_symbols = cfg.get('SYMBOLS', [])
    
    # 1. 데이터 준비
    if update_data:
        log("📥 최신 데이터 업데이트 중...")
        try:
            loader = MultiSymbolLoader()
            fetch_days = max(cfg.get('FETCH_DAYS', 2500), train_days + test_days + 365)
            for sym in target_symbols:
                df, path = loader.fetch_ohlcv(sym, days=fetch_days)
                if not df.empty:
                    df = loader.add_features(df)
                    df.to_csv(path)
        except Exception as e:
            return {"error": str(e)}

    # 2. 데이터 로드 및 정렬
    brain = Brain()
    full_dfs = {}
    valid_symbols = []
    global_min_date = None
    global_max_date = None

    for sym in target_symbols:
        df, _ = brain.load_data(sym)
        if df is None: continue
        df.sort_index(inplace=True)
        full_dfs[sym] = df
        valid_symbols.append(sym)
        
        if global_min_date is None or df.index[0] < global_min_date: global_min_date = df.index[0]
        if global_max_date is None or df.index[-1] > global_max_date: global_max_date = df.index[-1]

    if not valid_symbols: return {"error": "No valid data found"}

    # 3. 기간 설정
    min_req = timedelta(days=train_days)
    req_start = global_min_date + min_req
    if start_date:
        try:
            parsed = pd.Timestamp(start_date)
            if parsed > req_start: req_start = parsed
        except: pass
        
    req_end = global_max_date
    if end_date:
        try:
            parsed = pd.Timestamp(end_date)
            if parsed < global_max_date: req_end = parsed
        except: pass

    log(f"🚀 Real Portfolio Dynamic WFA 시작")
    log(f"📅 기간: {req_start.date()} ~ {req_end.date()}")

    # 4. Cerebro 설정
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(initial_cash)
    taker_fee = float(cfg.get('TAKER_FEE', 0.0005))
    cerebro.broker.set_slippage_perc(float(cfg.get('SLIPPAGE_PCT', 0.0002)))
    cerebro.broker.addcommissioninfo(FuturesComm(commission=taker_fee, leverage=float(cfg.get('LEVERAGE', 2.0))))

    # 5. 데이터 피드 추가 (Test 구간만 슬라이싱)
    total_len = 0
    for sym in valid_symbols:
        df = full_dfs[sym]
        mask = (df.index >= req_start) & (df.index <= req_end)
        test_df = df.loc[mask].copy()
        if len(test_df) > 0:
            feed = HMMData(dataname=test_df, name=sym)
            cerebro.adddata(feed)
            if len(test_df) > total_len: total_len = len(test_df)

    # 6. 콜백 래퍼
    def strategy_progress_wrapper(p):
        if progress_callback: progress_callback(p, 100, f"Portfolio... {p}%")

    # 7. 전략 추가
    cerebro.addstrategy(RealPortfolioStrategy,
                        config=cfg,
                        output_dir=None,
                        # [In-Strategy Params]
                        full_dfs=full_dfs,
                        train_days=train_days,
                        test_days=test_days,
                        brain=brain,
                        progress_callback=strategy_progress_wrapper,
                        total_steps=total_len
                        )

    # 8. 실행
    if progress_callback: progress_callback(5, 100, "Starting...") # 5% 시작
    results = cerebro.run()
    if progress_callback: progress_callback(100, 100, "Done")
    
    # 9. 결과 처리
    strat = results[0]
    final_val = cerebro.broker.getvalue()
    all_trades = strat.trade_records
    combined_stats = strat.daily_stats
    
    roi = (final_val - initial_cash) / initial_cash * 100
    
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    real_dir_name = f"RealPF_WFA_{ts}"
    real_dir_path = os.path.join(DATA_DIR, 'backtests', real_dir_name)
    os.makedirs(real_dir_path, exist_ok=True)
    
    zip_path = ""
    equity_curve = []
    
    try:
        # 포트폴리오 커브 저장
        if combined_stats:
            df_equity = pd.DataFrame(combined_stats)
            df_equity.drop_duplicates(subset=['Date'], keep='last', inplace=True)
            df_equity = df_equity[['Date', 'Total_Equity']]
            df_equity.rename(columns={'Date': 'time', 'Total_Equity': 'value'}, inplace=True)
            csv_path = os.path.join(real_dir_path, 'Portfolio_Curve.csv')
            df_equity.to_csv(csv_path, index=False)
            equity_curve = df_equity.to_dict(orient='records')
            
        # 통합 매매일지 저장
        if all_trades:
            df_trades = pd.DataFrame(all_trades)
            df_trades.sort_values(by='Entry_Time', inplace=True)
            df_trades.to_csv(os.path.join(real_dir_path, 'Master_Trade_Log.csv'), index=False)

        # 압축
        zip_base_name = os.path.join(DATA_DIR, 'backtests', real_dir_name)
        zip_path = shutil.make_archive(zip_base_name, 'zip', root_dir=os.path.join(DATA_DIR, 'backtests'), base_dir=real_dir_name)
        shutil.rmtree(real_dir_path)
    except Exception as e:
        log(f"⚠️ 결과 저장 오류: {e}")

    mdd = 0.0
    if equity_curve:
        df = pd.DataFrame(equity_curve)
        peak = df['value'].cummax()
        dd = (df['value'] - peak) / peak * 100
        mdd = abs(dd.min())

    log(f"🏁 Real Portfolio 완료. Final: ${final_val:.2f}, ROI: {roi:.2f}%")

    return {
        "type": "real_pf", 
        "portfolio_roi": round(roi, 2), 
        "avg_mdd": round(mdd, 2),
        "avg_roi": round(roi, 2), 
        "trade_count": len(all_trades), 
        "final_balance": round(final_val, 2),
        "csv_path": zip_path, 
        "equity_curve": equity_curve
    }