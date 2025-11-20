import sys
import os

# -----------------------------------------------------------
# [Linux Server Config]
# 서버에는 모니터가 없으므로, 차트 생성 시 화면 출력을 끕니다.
# 이 설정은 다른 모든 라이브러리(backtrader 등)보다 먼저 와야 합니다.
# -----------------------------------------------------------
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

import backtrader as bt
import pandas as pd
from datetime import datetime, timedelta

# 프로젝트 루트 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import CONFIG, DATA_DIR

# ---------------------------------------------------------
# 1. 데이터 피드 & 커미션
# ---------------------------------------------------------
class HMMData(bt.feeds.PandasData):
    lines = ('regime', 'rsi',)
    params = (('regime', -1), ('rsi', -1),)

class FuturesComm(bt.CommInfoBase):
    params = (('stocklike', False), ('commtype', bt.CommInfoBase.COMM_PERC), ('perc', 0.0005), ('leverage', 1.0),)
    def _getcredit(self, size, price): return super()._getcredit(size, price)
    def get_margin(self, price): return price / self.p.leverage

# ---------------------------------------------------------
# 2. 전략 클래스 (V5)
# ---------------------------------------------------------
class HMM_Pro_Strategy_V5(bt.Strategy):
    params = (('bull_id', None), ('bear_id', None), ('rsi_buy', 55), ('rsi_sell', 45), 
              ('ema_period', 20), ('trail_percent', 0.03), ('stop_loss', 0.05), 
              ('leverage', 1.0), ('debug', False))

    def __init__(self):
        self.regime = self.data.regime
        self.rsi = self.data.rsi
        self.ema = bt.indicators.EMA(self.data.close, period=self.p.ema_period)
        self.order = None
        self.entry_price = None 
        self.highest_price = None 
        self.lowest_price = None  

    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                self.entry_price = order.executed.price
                self.highest_price = order.executed.price
            elif order.issell():
                self.entry_price = order.executed.price
                self.lowest_price = order.executed.price
            self.order = None
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.order = None

    def next(self):
        if self.order: return
        
        current_regime = int(self.regime[0])
        current_rsi = self.rsi[0]
        current_price = self.data.close[0]
        current_ema = self.ema[0]
        pos_size = self.position.size
        
        portfolio_value = self.broker.getvalue()
        target_value = portfolio_value * 0.95 * self.p.leverage
        target_size = target_value / current_price

        # 1. 포지션 관리
        if pos_size != 0 and self.entry_price:
            if pos_size > 0: # Long
                pnl_pct = (current_price - self.entry_price) / self.entry_price
                if pnl_pct < -self.p.stop_loss: 
                    self.close(); return
                self.highest_price = max(self.highest_price, current_price)
                if (self.highest_price - current_price)/self.highest_price > self.p.trail_percent and current_price > self.entry_price: 
                    self.close(); return
            elif pos_size < 0: # Short
                pnl_pct = (self.entry_price - current_price) / self.entry_price
                if pnl_pct < -self.p.stop_loss: 
                    self.close(); return
                self.lowest_price = min(self.lowest_price, current_price)
                if (current_price - self.lowest_price)/self.lowest_price > self.p.trail_percent and current_price < self.entry_price: 
                    self.close(); return

        # 2. 국면 청산
        if pos_size > 0 and current_regime == self.p.bear_id: self.close(); return
        if pos_size < 0 and current_regime == self.p.bull_id: self.close(); return
        if pos_size != 0 and current_regime not in [self.p.bull_id, self.p.bear_id]: self.close(); return

        # 3. 진입
        if pos_size == 0:
            if current_regime == self.p.bull_id and current_rsi < self.p.rsi_buy and current_price > current_ema:
                self.order = self.order_target_size(target=target_size)
            elif current_regime == self.p.bear_id and current_rsi > self.p.rsi_sell and current_price < current_ema:
                self.order = self.order_target_size(target=-target_size)

# ---------------------------------------------------------
# 3. 디스코드 봇 전용 백테스팅 함수
# ---------------------------------------------------------
def run_single_backtest(symbol, days=365):
    clean_symbol = symbol.replace('/', '')
    file_path = os.path.join(DATA_DIR, f"{clean_symbol}_{CONFIG['TIMEFRAME']}.csv")
    
    if not os.path.exists(file_path):
        return f"❌ 데이터 파일 없음: {file_path}", None, None

    try:
        # 데이터 로드
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        start_date = datetime.now() - timedelta(days=days)
        df = df.loc[start_date:]
        
        if df.empty:
            return "❌ 해당 기간 데이터 없음", None, None

        # 국면 식별
        stats = df.groupby('Regime')['Log_Returns'].mean()
        bear_id = stats.idxmin()
        bull_id = stats.idxmax()

        # Cerebro 설정
        cerebro = bt.Cerebro()
        data = HMMData(dataname=df)
        cerebro.adddata(data)

        # 파라미터 설정
        is_btc = 'BTC' in symbol
        current_rsi_buy = 55 if is_btc else 50
        current_rsi_sell = 45 if is_btc else 50
        leverage = 2.0 if is_btc else 1.0
        
        comm_info = FuturesComm(commission=0.0005, leverage=leverage)
        cerebro.broker.addcommissioninfo(comm_info)
        
        cerebro.addstrategy(HMM_Pro_Strategy_V5, 
                            bull_id=bull_id, bear_id=bear_id,
                            rsi_buy=current_rsi_buy, rsi_sell=current_rsi_sell,
                            leverage=leverage)
        
        cerebro.broker.setcash(10000.0)
        cerebro.broker.set_shortcash(False)

        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')

        # 실행
        results = cerebro.run()
        strat = results[0]

        # 결과 처리
        portfolio_value = cerebro.broker.getvalue()
        roi = (portfolio_value - 10000.0) / 10000.0 * 100
        dd_info = strat.analyzers.drawdown.get_analysis()
        mdd = dd_info['max']['drawdown']
        
        trade_info = strat.analyzers.trades.get_analysis()
        total_trades = trade_info.total.total if 'total' in trade_info else 0
        won = trade_info.won.total if 'won' in trade_info and 'total' in trade_info.won else 0
        win_rate = (won / total_trades * 100) if total_trades > 0 else 0

        summary = (f"📊 **{symbol} 백테스트 결과 ({days}일)**\n"
                   f"💰 최종: ${portfolio_value:,.2f}\n"
                   f"📈 수익률: **{roi:.2f}%**\n"
                   f"🛡️ MDD: **{mdd:.2f}%**\n"
                   f"🎲 승률: {win_rate:.2f}% ({total_trades}회)\n"
                   f"⚙️ 레버리지: x{leverage}")

        # 엑셀 저장
        excel_path = f"report_{clean_symbol}.xlsx"
        report_data = {
            'Metric': ['Symbol', 'Days', 'Initial', 'Final', 'ROI', 'MDD', 'Win Rate', 'Trades'],
            'Value': [symbol, days, 10000, portfolio_value, f"{roi:.2f}%", f"{mdd:.2f}%", f"{win_rate:.2f}%", total_trades]
        }
        pd.DataFrame(report_data).to_excel(excel_path, index=False)

        # 차트 저장 (Agg 모드라 안전함)
        img_path = f"chart_{clean_symbol}.png"
        fig = cerebro.plot(style='candlestick', volume=False)[0][0]
        fig.set_size_inches(12, 8)
        fig.savefig(img_path, dpi=100)
        plt.close(fig)

        return summary, img_path, excel_path

    except Exception as e:
        return f"❌ 백테스트 오류: {e}", None, None