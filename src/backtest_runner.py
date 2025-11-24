import sys
import os
import shutil
import backtrader as bt
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

# 프로젝트 루트 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import CONFIG, DATA_DIR
from alpha_layer.brain import Brain
from data_layer.data_handler import MultiSymbolLoader

# --------------------------------------------------------- 
# 1. 데이터 피드 및 전략 클래스
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

class FuturesComm(bt.CommInfoBase):
    params = (
        ('stocklike', False), 
        ('commtype', bt.CommInfoBase.COMM_PERC),
        ('perc', 0.0005), 
        ('leverage', 1.0),
        ('slippage_perc', 0.0)
    )
    def _getcommission(self, size, price, pseudoexec=None, **kwargs):
        commission = super()._getcommission(size, price, pseudoexec, **kwargs)
        slippage_cost = abs(size) * price * self.p.slippage_perc
        return commission + slippage_cost
    def get_margin(self, price): 
        return price / self.p.leverage

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
        # [NEW] 펀딩비 설정 (기본 0.01%)
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
        
        self.last_exit_price = 0.0
        self.last_size = 0.0 
        
        self.trade_log = []
        self.daily_stats = []
        self.csv_path = ""

    def notify_order(self, order):
        if order.status in [order.Completed]:
            executed_price = order.executed.price
            executed_size = abs(order.executed.size)
            self.last_size = executed_size

            if order.isbuy():
                if self.position.size > 0: # Long Entry
                    self.entry_price = executed_price
                    self.stop_price = self.entry_price - (self.atr[0] * self.p.stop_atr)
                else: # Short Exit
                    self.last_exit_price = executed_price
            elif order.issell():
                if self.position.size < 0: # Short Entry
                    self.entry_price = executed_price
                    self.stop_price = self.entry_price + (self.atr[0] * self.p.stop_atr)
                else: # Long Exit
                    self.last_exit_price = executed_price
            
            self.order = None
            self.cooldown = 0
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed: return
        pnl_net = trade.pnlcomm
        pnl_gross = trade.pnl
        real_size = self.last_size if self.last_size > 0 else abs(trade.size)
        entry_val = real_size * trade.price
        pnl_pct = (pnl_net / entry_val) * 100 if entry_val > 0 else 0.0
        final_exit_price = self.last_exit_price if self.last_exit_price > 0 else (trade.price + (pnl_gross/real_size if real_size > 0 else 0))

        self.trade_log.append({
            'Symbol': self.data._name,
            'Type': 'Long' if trade.long else 'Short',
            'Entry_Date': bt.num2date(trade.dtopen).strftime('%Y-%m-%d %H:%M'),
            'Exit_Date': bt.num2date(trade.dtclose).strftime('%Y-%m-%d %H:%M'),
            'Entry_Price': trade.price,
            'Exit_Price': final_exit_price,
            'Size': real_size,
            'PnL': pnl_net,
            'ROI': pnl_pct,
            'Duration': trade.barlen
        })

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
        
        # [NEW] 2. 펀딩비 시뮬레이션 (8시간 주기)
        # 1시간봉 기준이므로 00, 08, 16시 정각에 차감
        dt = self.data.datetime.datetime(0)
        if dt.minute == 0 and dt.hour in [0, 8, 16]:
            if self.position.size != 0:
                # 포지션 가치 (Notional Value)
                pos_value = abs(self.position.size) * self.data.close[0]
                # 펀딩비 계산 (0.01% 가정)
                funding_cost = pos_value * self.p.funding_rate
                
                # 브로커 현금에서 직접 차감 (비용 처리)
                # add_cash는 백테스터의 가상 현금을 조절함
                self.broker.add_cash(-funding_cost)

        # 3. 매매 로직
        if self.order: return
        close = self.data.close[0]
        regime = int(self.regime[0])
        atr = self.atr[0]
        rsi = self.rsi[0]
        ema = self.ema[0]
        if self.cooldown > 0: self.cooldown -= 1

        if self.position.size != 0:
            if self.position.size > 0: # Long
                if self.data.low[0] < self.stop_price: self.close(); self.cooldown = self.p.cooldown_bars; return
                if close > self.entry_price + (atr * self.p.trail_trigger):
                    new_stop = close - (atr * self.p.trail_dist)
                    if new_stop > self.stop_price: self.stop_price = new_stop
                if regime == self.p.bear_id: self.close(); self.cooldown = self.p.cooldown_bars; return
            elif self.position.size < 0: # Short
                if self.data.high[0] > self.stop_price: self.close(); self.cooldown = self.p.cooldown_bars; return
                if close < self.entry_price - (atr * self.p.trail_trigger):
                    new_stop = close + (atr * self.p.trail_dist)
                    if new_stop < self.stop_price: self.stop_price = new_stop
                if regime == self.p.bull_id: self.close(); self.cooldown = self.p.cooldown_bars; return
        
        elif self.position.size == 0 and self.cooldown == 0:
            cash = self.broker.get_cash()
            if cash <= 0: return
            
            total_equity = self.broker.getvalue()
            allocation = total_equity / self.p.max_positions
            target_value = allocation * 0.95 * self.p.leverage
            size = target_value / close
            
            if size < 0.000001: return 

            if regime == self.p.bull_id:
                if close > ema and self.p.rsi_buy_lower < rsi < self.p.rsi_buy_upper:
                    self.order = self.buy(size=size)
            elif regime == self.p.bear_id:
                if close < ema and self.p.rsi_sell_lower < rsi < self.p.rsi_sell_upper:
                    self.order = self.sell(size=size)

    def stop(self):
        if self.daily_stats:
            df_stats = pd.DataFrame(self.daily_stats)
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            clean_sym = self.data._name.replace('/', '')
            filename = f"BT_{ts}_{clean_sym}.csv"
            
            if self.p.output_dir:
                save_dir = self.p.output_dir
            else:
                save_dir = os.path.join(DATA_DIR, 'backtests')
            
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
                
            full_path = os.path.join(save_dir, filename)
            df_stats.to_csv(full_path, index=False)
            self.csv_path = full_path

# --------------------------------------------------------- 
# 4. 실행 함수 (WFA Logic)
# --------------------------------------------------------- 
def run_walk_forward(symbol, initial_cash=10000.0, train_days=365, test_days=30, update_data=False, log_func=None, dynamic_config=None, output_dir=None):
    def log(msg):
        if log_func: log_func(msg)
        else: print(msg)

    cfg = dynamic_config if dynamic_config else CONFIG

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
        model, df_train_res = brain.train_model(df_train, symbol)
        regime_map, stats = brain.identify_regimes(df_train_res)
        sorted_stats = stats.sort_values()
        bear_id = sorted_stats.index[0]
        bull_id = sorted_stats.index[-1]

        test_features = df_test[['Log_Returns_Scaled', 'Range_Vol_Scaled', 'RSI_14_Scaled', 'OBV_Scaled']].values
        df_test['Regime'] = model.predict(test_features)
        
        cerebro = bt.Cerebro()
        data_feed = HMMData(dataname=df_test, name=symbol)
        cerebro.adddata(data_feed)
        
        base_leverage = float(cfg.get('LEVERAGE', 2.0))
        max_pos = int(cfg.get('MAX_OPEN_POSITIONS', 3))
        is_btc = 'BTC' in symbol
        lev = base_leverage if is_btc else 1.0 

        cerebro.addstrategy(HMM_Pro_Strategy_V5,
                            bull_id=bull_id, 
                            bear_id=bear_id, 
                            leverage=lev,
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
        
        cerebro.broker.setcash(current_cash)
        cerebro.broker.addcommissioninfo(FuturesComm(
            commission=float(cfg.get('COMMISSION', 0.0005)), 
            leverage=lev, 
            slippage_perc=float(cfg.get('SLIPPAGE_PCT', 0.0002))
        ))
        
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

    total_roi = (current_cash - initial_cash) / initial_cash * 100
    
    win_trades = [t for t in all_trade_logs if t['PnL'] > 0]
    win_rate = (len(win_trades) / len(all_trade_logs) * 100) if all_trade_logs else 0.0
    
    sorted_trades = sorted(all_trade_logs, key=lambda x: x['Exit_Date'])
    
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
    log("🚀 전체 포트폴리오 백테스팅 시작...")
    
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