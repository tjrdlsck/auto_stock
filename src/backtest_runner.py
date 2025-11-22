import sys
import os
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
    params = (('regime', -1),)

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
        ('atr_period', 14), ('rsi_period', 14), ('ema_period', 20),
        ('rsi_buy_upper', 65), ('rsi_buy_lower', 30),
        ('rsi_sell_lower', 35), ('rsi_sell_upper', 70),
        ('stop_atr', 2.0), ('trail_trigger', 2.0), ('trail_dist', 2.0),
        ('cooldown_bars', 2)
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
        
        # [FIX] 가격과 수량을 정확히 추적하기 위한 변수
        self.last_exit_price = 0.0
        self.last_size = 0.0 
        
        self.trade_log = []

    def notify_order(self, order):
        if order.status in [order.Completed]:
            executed_price = order.executed.price
            executed_size = abs(order.executed.size) # [FIX] 체결 수량 저장
            
            # 주문 체결 시 수량 업데이트 (진입/청산 모두 기록되지만, 청산 시 값이 최종 사용됨)
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
        
        # [FIX] trade.size 대신 notify_order에서 캡처한 정확한 size 사용
        real_size = self.last_size if self.last_size > 0 else abs(trade.size)
        
        # 진입 가치 계산
        entry_val = real_size * trade.price
        
        # 수익률 계산 (0 나누기 방지)
        if entry_val > 0:
            pnl_pct = (pnl_net / entry_val) * 100
        else:
            pnl_pct = 0.0
        
        # 청산 가격 결정
        final_exit_price = self.last_exit_price if self.last_exit_price > 0 else (trade.price + (pnl_gross/real_size if real_size > 0 else 0))

        self.trade_log.append({
            'Symbol': self.data._name,
            'Type': 'Long' if trade.long else 'Short',
            'Entry_Date': bt.num2date(trade.dtopen).strftime('%Y-%m-%d %H:%M'),
            'Exit_Date': bt.num2date(trade.dtclose).strftime('%Y-%m-%d %H:%M'),
            'Entry_Price': trade.price,
            'Exit_Price': final_exit_price,
            'Size': real_size, # 정확한 수량 저장
            'PnL': pnl_net,
            'ROI': pnl_pct,
            'Duration': trade.barlen
        })

    def next(self):
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
            target_value = cash * 0.95 * self.p.leverage
            size = target_value / close
            
            if size < 0.000001: return # 최소 수량 제한

            if regime == self.p.bull_id:
                if close > ema and self.p.rsi_buy_lower < rsi < self.p.rsi_buy_upper:
                    self.order = self.buy(size=size)
            elif regime == self.p.bear_id:
                if close < ema and self.p.rsi_sell_lower < rsi < self.p.rsi_sell_upper:
                    self.order = self.sell(size=size)

# --------------------------------------------------------- 
# 4. 실행 함수 (WFA Logic)
# --------------------------------------------------------- 
def run_walk_forward(symbol, initial_cash=10000.0, train_days=365, test_days=30, update_data=False, log_func=None):
    def log(msg):
        if log_func: log_func(msg)
        else: print(msg)

    if update_data:
        log(f"📥 [{symbol}] 최신 데이터 다운로드 및 갱신 중...")
        try:
            loader = MultiSymbolLoader()
            required_days = train_days + test_days + 100
            fetch_days = max(CONFIG['FETCH_DAYS'], required_days)
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
        
        is_btc = 'BTC' in symbol
        lev = 2.0 if is_btc else 1.0
        
        cerebro.addstrategy(HMM_Pro_Strategy_V5,
                            bull_id=bull_id, bear_id=bear_id, leverage=lev)
        
        cerebro.broker.setcash(current_cash)
        cerebro.broker.addcommissioninfo(FuturesComm(
            commission=CONFIG['COMMISSION'], leverage=lev, slippage_perc=CONFIG['SLIPPAGE_PCT']
        ))
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        
        log(f"🚀 [{symbol}] 시뮬레이션 실행 (기간: {test_days}일)...")
        results = cerebro.run()
        strat = results[0]
        current_cash = cerebro.broker.getvalue()
        
        if strat.trade_log:
            all_trade_logs = strat.trade_log

    except Exception as e:
        log(f"❌ 시뮬레이션 에러: {e}")
        return {"error": str(e)}

    total_roi = (current_cash - initial_cash) / initial_cash * 100
    
    running_pnl = 0
    sorted_trades = sorted(all_trade_logs, key=lambda x: x['Exit_Date'])
    
    equity_curve.append({"time": df_test.index[0].strftime('%Y-%m-%d'), "value": initial_cash})
    for t in sorted_trades:
        running_pnl += t['PnL']
        equity_curve.append({
            "time": t['Exit_Date'][:10],
            "value": initial_cash + running_pnl
        })
    equity_curve.append({"time": df_test.index[-1].strftime('%Y-%m-%d'), "value": current_cash})

    log(f"✅ [{symbol}] 완료: 수익률 {total_roi:+.2f}%, 거래 {len(all_trade_logs)}회")

    return {
        "symbol": symbol,
        "roi": round(total_roi, 2),
        "final_balance": round(current_cash, 2),
        "trade_count": len(all_trade_logs),
        "trades": sorted_trades,
        "equity_curve": equity_curve
    }

def run_batch_backtest(initial_cash=10000.0, train_days=365, test_days=30, update_data=False, log_func=None):
    def log(msg):
        if log_func: log_func(msg)
        else: print(msg)
    results = []
    log("🚀 전체 포트폴리오 백테스팅 시작...")
    for symbol in CONFIG['SYMBOLS']:
        res = run_walk_forward(symbol, initial_cash, train_days, test_days, update_data, log_func)
        if "error" not in res: results.append(res)
    
    if not results: return {"error": "No results"}
    avg_roi = sum(r['roi'] for r in results) / len(results)
    total_balance = sum(r['final_balance'] for r in results)
    start_total = initial_cash * len(results)
    pf_roi = (total_balance - start_total) / start_total * 100
    log(f"🏆 [종합] 평균 ROI: {avg_roi:+.2f}%, 전체 ROI: {pf_roi:+.2f}%")
    return {"type": "batch", "avg_roi": round(avg_roi, 2), "portfolio_roi": round(pf_roi, 2), "details": results}