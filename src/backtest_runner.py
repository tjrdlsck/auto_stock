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
        ('bull_id', None), ('bear_id', None), # 이제 사용하지 않거나, 외부 매핑 확인용으로 사용
        ('regime_map', {}), # {0: 'bear', 1: 'sideways', 2: 'bull'} 형태
        ('leverage', 2.0),
        ('max_positions', 3),
        ('rsi_buy_upper', 65), ('rsi_buy_lower', 30),
        ('rsi_sell_lower', 35), ('rsi_sell_upper', 70),
        ('stop_atr', 2.0), ('trail_trigger', 2.0), ('trail_dist', 2.0),
        ('cooldown_bars', 2),
        ('output_dir', None),
        ('funding_rate', 0.0001) 
    )

    def __init__(self):
        # [수정] 내부 계산 없이 데이터 피드의 값 직접 참조 (실전 정합성 확보)
        self.regime = self.data.regime
        self.atr = self.data.atr
        self.rsi = self.data.rsi
        self.ema = self.data.ema
        
        self.order = None
        self.entry_price = 0.0
        self.stop_price = 0.0
        self.cooldown = 0
        
        # 메타데이터 기록용
        self.entry_metadata = {} 
        self.exit_reason = "Manual/End" 
        self.trade_log = []   
        self.daily_stats = [] 

    def notify_order(self, order):
        if order.status in [order.Completed]:
            executed_price = order.executed.price
            executed_size = abs(order.executed.size)

            if order.isbuy():
                if self.position.size > 0: 
                    self.entry_price = executed_price
                    self.stop_price = self.entry_price - (self.atr[0] * self.p.stop_atr)
                    self.entry_metadata = {
                        'size': executed_size,
                        'entry_time': self.data.datetime.datetime(0),
                        'regime': 'Bull' # Long은 Bull에서만 진입 가정
                    }
            elif order.issell():
                if self.position.size < 0:
                    self.entry_price = executed_price
                    self.stop_price = self.entry_price + (self.atr[0] * self.p.stop_atr)
                    self.entry_metadata = {
                        'size': executed_size,
                        'entry_time': self.data.datetime.datetime(0),
                        'regime': 'Bear' # Short는 Bear에서만 진입 가정
                    }
            
            self.order = None
            self.cooldown = 0
            
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed: return
        
        real_size = self.entry_metadata.get('size', 0.0)
        if real_size == 0 and trade.history:
            real_size = abs(trade.history[0].event.size)
            
        entry_time = self.entry_metadata.get('entry_time', bt.num2date(trade.dtopen))
        entry_regime = self.entry_metadata.get('regime', 'Unknown')
        
        pnl_net = trade.pnlcomm
        
        # ROI 계산
        safe_leverage = self.p.leverage if self.p.leverage > 0 else 1.0
        entry_val = (trade.price * real_size) / safe_leverage
        roi_pct = 0.0
        if entry_val != 0:
            roi_pct = (pnl_net / entry_val) * 100

        exit_time = bt.num2date(trade.dtclose)
        duration = exit_time - entry_time

        exit_price_approx = trade.price
        if real_size > 0:
            if trade.long:
                # Long: PnL이 양수면 Exit > Entry
                exit_price_approx = trade.price + (trade.pnl / real_size)
            else:
                # Short: PnL이 양수면 Exit < Entry (Entry에서 빼야 함)
                exit_price_approx = trade.price - (trade.pnl / real_size)

        self.trade_log.append({
            'Symbol': self.data._name,
            'Type': 'Long' if trade.long else 'Short',
            'Entry_Time': entry_time,
            'Exit_Time': exit_time,
            'Duration_Hours': duration.total_seconds() / 3600,
            'Entry_Price': trade.price,
            'Exit_Price': exit_price_approx, # [수정됨] 계산된 값 사용
            'Size': real_size,
            'PnL': pnl_net,
            'Fee': trade.commission,
            'ROI': roi_pct,
            'Entry_Regime': entry_regime,
            'Exit_Reason': self.exit_reason
        })
        self.exit_reason = "Manual/End" 

    def next(self):
        # 1. 일별 통계
        self.daily_stats.append({
            'Date': self.data.datetime.datetime(0),
            'Close': self.data.close[0],
            'Regime': int(self.regime[0]),
            'Position_Size': self.position.size,
            'Portfolio_Value': self.broker.getvalue()
        })
        
        # Funding Fee (간이)
        dt = self.data.datetime.datetime(0)
        if dt.minute == 0 and dt.hour in [0, 8, 16]:
            if self.position.size != 0:
                pos_value = abs(self.position.size) * self.data.close[0]
                self.broker.add_cash(-pos_value * self.p.funding_rate)

        if self.order: return
        
        close = self.data.close[0]
        # [수정] Regime은 Brain에서 예측하여 주입한 값을 그대로 사용
        # regime_map: {0: 'bear', 1: 'sideways', 2: 'bull'} (예시)
        # self.regime[0]은 0, 1, 2 중 하나
        
        current_regime_id = int(self.regime[0])
        
        # regime_map을 통해 현재 국면의 의미 파악
        # (Runner에서 주입해줘야 함. 여기서는 bull_id/bear_id params 사용)
        is_bull = (current_regime_id == self.p.bull_id)
        is_bear = (current_regime_id == self.p.bear_id)
        
        atr = self.atr[0]
        rsi = self.rsi[0]
        ema = self.ema[0]
        
        if self.cooldown > 0: self.cooldown -= 1

        # [청산 로직] (기존과 동일하되 변수명만 self.atr 등으로 변경)
        if self.position.size != 0:
            should_close = False
            reason = ""
            
            if self.position.size > 0: # Long
                if self.data.low[0] < self.stop_price: 
                    should_close = True; reason = "StopLoss"
                elif close > self.entry_price + (atr * self.p.trail_trigger):
                    new_stop = close - (atr * self.p.trail_dist)
                    if new_stop > self.stop_price: self.stop_price = new_stop
                elif is_bear: # 국면 전환
                    should_close = True; reason = "RegimeChange(Bear)"
                    
            elif self.position.size < 0: # Short
                if self.data.high[0] > self.stop_price: 
                    should_close = True; reason = "StopLoss"
                elif close < self.entry_price - (atr * self.p.trail_trigger):
                    new_stop = close + (atr * self.p.trail_dist)
                    if new_stop < self.stop_price: self.stop_price = new_stop
                elif is_bull: # 국면 전환
                    should_close = True; reason = "RegimeChange(Bull)"
            
            if should_close:
                self.exit_reason = reason
                self.close()
                self.cooldown = self.p.cooldown_bars
                return
        
        # [진입 로직]
        elif self.position.size == 0 and self.cooldown == 0:
            cash = self.broker.get_cash()
            if cash <= 0: return
            
            # 자금 관리
            max_pos = self.p.max_positions if self.p.max_positions > 0 else 1
            allocation = self.broker.getvalue() / max_pos
            target_value = allocation * 0.95 * self.p.leverage
            size = target_value / close if close > 0 else 0
            
            if size < 0.000001: return 

            if is_bull:
                if close > ema and self.p.rsi_buy_lower < rsi < self.p.rsi_buy_upper:
                    self.order = self.buy(size=size)
            elif is_bear:
                if close < ema and self.p.rsi_sell_lower < rsi < self.p.rsi_sell_upper:
                    self.order = self.sell(size=size)
    def stop(self):
        """
        전략 실행 종료 시 호출되는 메서드.
        수집된 로그 데이터를 CSV 파일로 저장합니다.
        """
        save_dir = self.p.output_dir
        
        # 저장 경로가 없으면(Walk-Forward 중간 과정 등) 저장하지 않음
        if not save_dir: return

        # 1. Master Trade Log (상세 매매 기록 저장)
        # 변수명 수정: self.trade_records -> self.trade_log
        if self.trade_log:
            try:
                df_trades = pd.DataFrame(self.trade_log)
                # 보기 좋게 시간순 정렬
                if 'Entry_Time' in df_trades.columns:
                    df_trades.sort_values(by='Entry_Time', inplace=True)
                
                # CSV 저장
                save_path = os.path.join(save_dir, 'Master_Trade_Log.csv')
                df_trades.to_csv(save_path, index=False)
            except Exception as e:
                print(f"⚠️ Trade Log 저장 실패: {e}")
            
        # 2. Daily Portfolio Stats (일별 자산 흐름 저장)
        if self.daily_stats:
            try:
                df_stats = pd.DataFrame(self.daily_stats)
                # 중복 데이터 정리 (하루의 마지막 상태만 남김)
                df_stats.drop_duplicates(subset=['Date'], keep='last', inplace=True)
                
                # CSV 저장
                save_path = os.path.join(save_dir, 'Daily_Portfolio_Stats.csv')
                df_stats.to_csv(save_path, index=False)
            except Exception as e:
                print(f"⚠️ Daily Stats 저장 실패: {e}")


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
            if trade.long:
                # Long: Exit = Entry + (PnL / Size)
                exit_price_approx = trade.price + (trade.pnl / real_size)
            else:
                # Short: PnL = (Entry - Exit) * Size  => Exit = Entry - (PnL / Size)
                exit_price_approx = trade.price - (trade.pnl / real_size)

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
# [수정] 날짜 지정 기능이 포함된 Walk-Forward 엔진
def run_walk_forward(symbol, initial_cash=10000.0, train_days=1500, test_days=30, 
                     start_date=None, end_date=None, 
                     update_data=False, log_func=None, dynamic_config=None, output_dir=None):
    def log(msg):
        if log_func: log_func(msg)
        else: print(msg)

    # [안전장치 1] 입력값 유효성 검사
    if test_days <= 0:
        return {"error": f"재학습 주기(test_days)는 0일 수 없습니다. 최소 1일 이상 설정해주세요."}
    
    cfg = dynamic_config if dynamic_config else CONFIG
    min_train_days = int(cfg.get('MIN_TRAIN_DAYS', 30))

    # 데이터 업데이트 및 로드
    if update_data:
        log(f"📥 [{symbol}] Walk-Forward용 데이터 준비 중...")
        try:
            loader = MultiSymbolLoader()
            fetch_days = max(cfg.get('FETCH_DAYS', 2500), train_days + test_days + 365)
            df, path = loader.fetch_ohlcv(symbol, days=fetch_days)
            if not df.empty:
                df = loader.add_features(df)
                df.to_csv(path)
            else:
                return {"error": "Data fetch failed"}
        except Exception as e:
            return {"error": str(e)}

    brain = Brain()
    df_full, _ = brain.load_data(symbol)
    if df_full is None or df_full.empty:
        return {"error": "Data load failed"}
    
    df_full.sort_index(inplace=True)
    
    # 시간 기반 연산 설정
    data_min_date = df_full.index[0]
    data_max_date = df_full.index[-1]
    train_td = timedelta(days=train_days)
    test_td = timedelta(days=test_days)

    if (data_max_date - data_min_date) < train_td:
         msg = f"⚠️ 데이터 부족: 보유 기간 {(data_max_date - data_min_date).days}일 vs 요구 학습 기간 {train_days}일"
         log(msg)
         return {"error": msg}

    # 시작/종료 시점 설정
    default_start_date = data_min_date + train_td
    req_start_date = default_start_date
    if start_date:
        try:
            parsed_start = pd.Timestamp(start_date)
            if parsed_start < default_start_date:
                log(f"⚠️ 요청한 시작일({parsed_start.date()})은 학습 데이터 부족으로 인해 {default_start_date.date()}로 조정되었습니다.")
                req_start_date = default_start_date
            else:
                req_start_date = parsed_start
        except: return {"error": f"Invalid start_date: {start_date}"}

    req_end_date = data_max_date
    if end_date:
        try:
            parsed_end = pd.Timestamp(end_date)
            if parsed_end < data_max_date: req_end_date = parsed_end
        except: return {"error": f"Invalid end_date: {end_date}"}
            
    if req_start_date >= req_end_date:
         return {"error": "설정된 기간이 유효하지 않습니다. (Start >= End)"}

    total_duration = req_end_date - req_start_date
    estimated_steps = int(total_duration / test_td) + 1
    
    log(f"🚀 [{symbol}] Walk-Forward 시작 (Overlap 방지 적용)")
    log(f"📅 검증 기간: {req_start_date.date()} ~ {req_end_date.date()} (약 {estimated_steps}회 반복)")
    
    current_cash = initial_cash
    all_trades = []
    equity_curve = []
    iteration = 0
    current_date = req_start_date
    
    while current_date < req_end_date:
        test_start_ts = current_date
        test_end_ts = min(current_date + test_td, req_end_date)
        
        if test_end_ts <= test_start_ts: break
        
        train_start_ts = test_start_ts - train_td
        
        # [Phase 7 수정] 경계 중복 방지를 위한 불리언 마스킹 (미만 연산자 사용)
        # Train: [Start, Test_Start)
        mask_train = (df_full.index >= train_start_ts) & (df_full.index < test_start_ts)
        train_data = df_full.loc[mask_train].copy()
        
        # Test: [Test_Start, Test_End)  <-- 기존 <= 에서 < 로 변경됨
        mask_test = (df_full.index >= test_start_ts) & (df_full.index < test_end_ts)
        test_data = df_full.loc[mask_test].copy()
        
        if len(train_data) < 100 or len(test_data) == 0:
            current_date = test_end_ts
            continue
        
        iteration += 1
        period_str = f"{test_data.index[0].date()}~{test_data.index[-1].date()}"
        log(f"🔄 [Step {iteration}/{estimated_steps}] AI 재학습 및 검증 ({period_str})")

        try:
            model, _ = brain.train_model(train_data, symbol)
            regime_map, stats = brain.identify_regimes(train_data)
            
            bull_id = [k for k, v in regime_map.items() if 'Bull' in v][0]
            bear_id = [k for k, v in regime_map.items() if 'Bear' in v][0]
            
            X_test = test_data[['Log_Returns_Scaled', 'Range_Vol_Scaled', 'RSI_14_Scaled', 'OBV_Scaled']].values
            test_data['Regime'] = model.predict(X_test)
            
        except Exception:
            current_date = test_end_ts
            continue

        cerebro = bt.Cerebro()
        cerebro.broker.setcash(current_cash)
        
        taker_fee = float(cfg.get('TAKER_FEE', 0.0005))
        cerebro.broker.set_slippage_perc(float(cfg.get('SLIPPAGE_PCT', 0.0002)))
        cerebro.broker.addcommissioninfo(FuturesComm(commission=taker_fee, leverage=float(cfg.get('LEVERAGE', 2.0))))
        
        data_feed = HMMData(dataname=test_data, name=symbol)
        cerebro.adddata(data_feed)
        
        cerebro.addstrategy(HMM_Pro_Strategy_V5,
                            bull_id=bull_id, bear_id=bear_id,
                            leverage=float(cfg.get('LEVERAGE', 2.0)),
                            max_positions=int(cfg.get('MAX_OPEN_POSITIONS', 3)),
                            rsi_buy_upper=float(cfg.get('RSI_BUY_UPPER', 65)),
                            rsi_buy_lower=float(cfg.get('RSI_BUY_LOWER', 30)),
                            rsi_sell_lower=float(cfg.get('RSI_SELL_LOWER', 35)),
                            rsi_sell_upper=float(cfg.get('RSI_SELL_UPPER', 70)),
                            stop_atr=float(cfg.get('STOP_LOSS_ATR', 2.0)),
                            trail_trigger=float(cfg.get('TRAIL_TRIGGER_ATR', 2.0)),
                            trail_dist=float(cfg.get('TRAIL_DIST_ATR', 2.0)),
                            output_dir=None
                            )

        results = cerebro.run()
        strat = results[0]
        
        current_cash = cerebro.broker.getvalue()
        if strat.trade_log:
            all_trades.extend(strat.trade_log)
        
        equity_curve.append({
            "time": test_data.index[-1].strftime('%Y-%m-%d'),
            "value": current_cash
        })
        
        current_date = test_end_ts

    total_roi = 0.0
    if initial_cash > 0:
        total_roi = (current_cash - initial_cash) / initial_cash * 100

    win_trades = [t for t in all_trades if t['PnL'] > 0]
    win_rate = (len(win_trades) / len(all_trades) * 100) if all_trades else 0.0
    
    df_trades = pd.DataFrame(all_trades)
    mdd = 0.0
    csv_save_path = ""
    
    if output_dir:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_save_path = os.path.join(output_dir, f"WFA_{symbol.replace('/','')}_{ts}.csv")
        if not df_trades.empty:
            df_trades.sort_values(by='Entry_Time', inplace=True)
            df_trades.to_csv(csv_save_path, index=False)
            df_trades['cum_pnl'] = df_trades['PnL'].cumsum()
            df_trades['equity'] = initial_cash + df_trades['cum_pnl']
            peak = df_trades['equity'].cummax()
            dd = (df_trades['equity'] - peak) / peak * 100
            mdd = abs(dd.min())

    log(f"🏁 [{symbol}] 완료. Cash: ${current_cash:.2f}, ROI: {total_roi:.2f}%")

    return {
        "symbol": symbol, "roi": round(total_roi, 2), "mdd": round(mdd, 2),
        "win_rate": round(win_rate, 2), "final_balance": round(current_cash, 2),
        "trade_count": len(all_trades), "trades": all_trades,
        "equity_curve": equity_curve, "csv_path": csv_save_path, "params": cfg
    }

def run_batch_backtest(initial_cash=10000.0, train_days=365, test_days=30, 
                       start_date=None, end_date=None, # [Phase 4] 날짜 인자 추가
                       update_data=False, log_func=None, dynamic_config=None):
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
        # [Phase 4] start_date, end_date 전달 추가
        res = run_walk_forward(symbol, initial_cash, train_days, test_days, 
                               start_date=start_date, end_date=end_date, 
                               update_data=update_data, log_func=log_func, 
                               dynamic_config=cfg, output_dir=batch_dir_path)
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
        
        # 데이터프레임 병합 시 빈 날짜 처리 강화
        if df_list:
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
                                update_data=False, log_func=None, dynamic_config=None):
    def log(msg):
        if log_func: log_func(msg)
        else: print(msg)

    if test_days <= 0:
        return {"error": f"재학습 주기(test_days)는 0일 수 없습니다."}

    cfg = dynamic_config if dynamic_config else CONFIG
    target_symbols = cfg.get('SYMBOLS', [])
    min_train_days = int(cfg.get('MIN_TRAIN_DAYS', 30))
    
    log(f"🚀 Real Portfolio WFA 시작 (Symbols: {len(target_symbols)}개)")

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

    # 데이터 로드
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

    if not valid_symbols:
        return {"error": "No valid data found"}

    # 시간 설정 (Timedelta)
    train_td = timedelta(days=train_days)
    test_td = timedelta(days=test_days)
    
    # 시작/종료 시점 설정
    default_start_date = global_min_date + train_td
    req_start_date = default_start_date
    if start_date:
        try:
            parsed_start = pd.Timestamp(start_date)
            if parsed_start < default_start_date: 
                req_start_date = default_start_date
            else: 
                req_start_date = parsed_start 
        except: return {"error": "Invalid start_date"}

    req_end_date = global_max_date
    if end_date:
        try:
            parsed_end = pd.Timestamp(end_date)
            if parsed_end < global_max_date: req_end_date = parsed_end
        except: return {"error": "Invalid end_date"}

    if req_start_date >= req_end_date:
        return {"error": "설정된 기간이 유효하지 않습니다."}

    total_duration = req_end_date - req_start_date
    estimated_steps = int(total_duration / test_td) + 1
    log(f"📅 검증 기간: {req_start_date.date()} ~ {req_end_date.date()} (약 {estimated_steps}회 반복)")

    current_cash = initial_cash
    all_trades = []
    combined_stats = []
    iteration = 0
    current_date = req_start_date

    while current_date < req_end_date:
        test_start_ts = current_date
        test_end_ts = min(current_date + test_td, req_end_date)
        
        if test_end_ts <= test_start_ts: break
        
        train_start_ts = test_start_ts - train_td
        
        iteration += 1
        log(f"🔄 [Step {iteration}/{estimated_steps}] Portfolio 재학습 및 검증 ({test_start_ts.date()} ~ {test_end_ts.date()})")
        
        step_regime_map = {}
        step_feeds = {}
        
        for sym in valid_symbols:
            df = full_dfs[sym]
            
            # [Phase 7 수정] 경계 중복 방지를 위한 불리언 마스킹 (미만 연산자 사용)
            # Train: [Start, Test_Start)
            mask_train = (df.index >= train_start_ts) & (df.index < test_start_ts)
            train_data = df.loc[mask_train].copy()
            
            # Test: [Test_Start, Test_End)
            mask_test = (df.index >= test_start_ts) & (df.index < test_end_ts)
            test_data = df.loc[mask_test].copy()
            
            if len(train_data) < 100 or len(test_data) == 0: continue

            try:
                model, _ = brain.train_model(train_data, sym)
                r_map, _ = brain.identify_regimes(train_data)
                
                bull_idx = [k for k, v in r_map.items() if 'Bull' in v][0]
                bear_idx = [k for k, v in r_map.items() if 'Bear' in v][0]
                step_regime_map[sym] = {'bull': bull_idx, 'bear': bear_idx}
                
                X_test = test_data[['Log_Returns_Scaled', 'Range_Vol_Scaled', 'RSI_14_Scaled', 'OBV_Scaled']].values
                test_data['Regime'] = model.predict(X_test)
                step_feeds[sym] = HMMData(dataname=test_data, name=sym)
            except: continue

        if not step_feeds:
            current_date = test_end_ts
            continue

        cerebro = bt.Cerebro()
        cerebro.broker.setcash(current_cash)
        
        taker_fee = float(cfg.get('TAKER_FEE', 0.0005))
        cerebro.broker.set_slippage_perc(float(cfg.get('SLIPPAGE_PCT', 0.0002)))
        cerebro.broker.addcommissioninfo(FuturesComm(commission=taker_fee, leverage=float(cfg.get('LEVERAGE', 2.0))))
        
        for sym, feed in step_feeds.items():
            cerebro.adddata(feed)
            
        cerebro.addstrategy(RealPortfolioStrategy, regime_map=step_regime_map, config=cfg, output_dir=None)

        try:
            results = cerebro.run()
            strat = results[0]
            current_cash = cerebro.broker.getvalue()
            
            if hasattr(strat, 'trade_records'): all_trades.extend(strat.trade_records)
            if hasattr(strat, 'daily_stats'): combined_stats.extend(strat.daily_stats)
        except Exception as e:
            log(f"❌ [Step {iteration}] 에러: {e}")
            
        current_date = test_end_ts

    roi = 0.0
    if initial_cash > 0: roi = (current_cash - initial_cash) / initial_cash * 100
        
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    real_dir_name = f"RealPF_WFA_{ts}"
    real_dir_path = os.path.join(DATA_DIR, 'backtests', real_dir_name)
    os.makedirs(real_dir_path, exist_ok=True)
    
    zip_path = ""
    equity_curve = []
    
    try:
        if combined_stats:
            df_equity = pd.DataFrame(combined_stats)
            df_equity.drop_duplicates(subset=['Date'], keep='last', inplace=True)
            df_equity = df_equity[['Date', 'Total_Equity']]
            df_equity.rename(columns={'Date': 'time', 'Total_Equity': 'value'}, inplace=True)
            
            csv_path = os.path.join(real_dir_path, 'Portfolio_Curve.csv')
            df_equity.to_csv(csv_path, index=False)
            equity_curve = df_equity.to_dict(orient='records')
            
        if all_trades:
            df_trades = pd.DataFrame(all_trades)
            df_trades.sort_values(by='Entry_Time', inplace=True)
            df_trades.to_csv(os.path.join(real_dir_path, 'Master_Trade_Log.csv'), index=False)

        zip_base_name = os.path.join(DATA_DIR, 'backtests', real_dir_name)
        zip_path = shutil.make_archive(zip_base_name, 'zip', root_dir=os.path.join(DATA_DIR, 'backtests'), base_dir=real_dir_name)
        shutil.rmtree(real_dir_path)
    except Exception as e:
        log(f"⚠️ 결과 저장 중 오류: {e}")

    mdd = 0.0
    if equity_curve:
        df = pd.DataFrame(equity_curve)
        peak = df['value'].cummax()
        peak = peak.replace(0, 1) 
        dd = (df['value'] - peak) / peak * 100
        mdd = abs(dd.min())

    log(f"🏁 Real Portfolio WFA 완료. Final: ${current_cash:.2f}, ROI: {roi:.2f}%")

    return {
        "type": "real_pf",
        "portfolio_roi": round(roi, 2),
        "avg_mdd": round(mdd, 2),
        "avg_roi": round(roi, 2),
        "trade_count": len(all_trades),
        "final_balance": round(current_cash, 2),
        "csv_path": zip_path,
        "equity_curve": equity_curve
    }