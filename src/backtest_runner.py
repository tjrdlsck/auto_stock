import sys
import os

# -----------------------------------------------------------
# [Linux Server Config]
# 서버에는 모니터가 없으므로, 차트 생성 시 화면 출력을 끕니다.
# 이 설정은 다른 모든 라이브러리(backtrader 등)보다 먼저 와야 합니다.
# -----------------------------------------------------------
import backtrader as bt
import pandas as pd
from datetime import datetime, timedelta
import os

# 프로젝트 루트 경로 추가 (sys.path is already handled by the calling script, but good for standalone running)
# Note: The original file had a sys.path modification, which is kept here for context,
# but might be redundant depending on execution context.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import CONFIG, DATA_DIR

# ---------------------------------------------------------
# 1. 데이터 피드 & 커미션 (유지)
# ---------------------------------------------------------
class HMMData(bt.feeds.PandasData):
    lines = ('regime', 'rsi',)
    params = (('regime', -1), ('rsi', -1),)

class FuturesComm(bt.CommInfoBase):
    params = (('stocklike', False), ('commtype', bt.CommInfoBase.COMM_PERC), ('perc', 0.0005), ('leverage', 1.0),)
    def _getcredit(self, size, price): return super()._getcredit(size, price)
    def get_margin(self, price): return price / self.p.leverage

# ---------------------------------------------------------
# 2. 전략 클래스 (V5) (유지)
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

        if pos_size != 0 and self.entry_price:
            if pos_size > 0:
                pnl_pct = (current_price - self.entry_price) / self.entry_price
                if pnl_pct < -self.p.stop_loss: self.close(); return
                self.highest_price = max(self.highest_price, current_price)
                if (self.highest_price - current_price)/self.highest_price > self.p.trail_percent and current_price > self.entry_price: self.close(); return
            elif pos_size < 0:
                pnl_pct = (self.entry_price - current_price) / self.entry_price
                if pnl_pct < -self.p.stop_loss: self.close(); return
                self.lowest_price = min(self.lowest_price, current_price)
                if (current_price - self.lowest_price)/self.lowest_price > self.p.trail_percent and current_price < self.entry_price: self.close(); return

        if pos_size > 0 and current_regime == self.p.bear_id: self.close(); return
        if pos_size < 0 and current_regime == self.p.bull_id: self.close(); return
        if pos_size != 0 and current_regime not in [self.p.bull_id, self.p.bear_id]: self.close(); return

        if pos_size == 0:
            if current_regime == self.p.bull_id and current_rsi < self.p.rsi_buy and current_price > self.ema:
                self.order = self.order_target_size(target=target_size)
            elif current_regime == self.p.bear_id and current_rsi > self.p.rsi_sell and current_price < self.ema:
                self.order = self.order_target_size(target=-target_size)

# ---------------------------------------------------------
# 3. 디스코드 봇 전용 백테스팅 함수 (수정됨)
# ---------------------------------------------------------
def run_single_backtest(symbol, days=365):
    """차트/엑셀 없이 텍스트 요약만 반환"""
    clean_symbol = symbol.replace('/', '')
    file_path = os.path.join(DATA_DIR, f"{clean_symbol}_{CONFIG['TIMEFRAME']}.csv")
    
    if not os.path.exists(file_path):
        return f"❌ 데이터 파일 없음: {clean_symbol}", None

    try:
        # 데이터 로드 및 기간 필터링
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        start_date = datetime.now() - timedelta(days=days)
        df = df.loc[start_date:]
        
        if df.empty: return f"❌ 데이터 부족 ({days}일)", None

        # 국면 식별
        stats = df.groupby('Regime')['Log_Returns'].mean()
        bear_id = stats.idxmin()
        bull_id = stats.idxmax()

        # Cerebro 설정
        cerebro = bt.Cerebro()
        data = HMMData(dataname=df)
        cerebro.adddata(data)
        
        # 전략 및 커미션 설정 (기존과 동일)
        is_btc = 'BTC' in symbol
        cerebro.addstrategy(HMM_Pro_Strategy_V5, 
                            bull_id=bull_id, bear_id=bear_id,
                            rsi_buy=55 if is_btc else 50, 
                            rsi_sell=45 if is_btc else 50,
                            leverage=2.0 if is_btc else 1.0)
        
        cerebro.broker.setcash(10000.0)
        cerebro.broker.addcommissioninfo(FuturesComm(commission=0.0005, leverage=2.0 if is_btc else 1.0))
        
        # 분석기
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')

        # 실행
        results = cerebro.run()
        strat = results[0]

        # 결과 계산
        final_val = cerebro.broker.getvalue()
        roi = (final_val - 10000.0) / 10000.0 * 100
        dd_info = strat.analyzers.drawdown.get_analysis()
        mdd = dd_info['max']['drawdown']
        
        trade_info = strat.analyzers.trades.get_analysis()
        total_trades = trade_info.total.total if 'total' in trade_info else 0
        won = trade_info.won.total if 'won' in trade_info and 'total' in trade_info.won else 0
        win_rate = (won / total_trades * 100) if total_trades > 0 else 0

        # 텍스트 요약 생성 (마크다운 활용)
        summary = (
            f"📊 **{symbol} 백테스트 ({days}일)**\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"💰 수익률: **{roi:+.2f}%**\n"
            f"🛡️ MDD: **{mdd:.2f}%**\n"
            f"🎲 승률: **{win_rate:.1f}%** ({total_trades}회)\n"
            f"💵 최종: ${final_val:,.0f}"
        )
        
        # 배치 처리를 위해 수치 데이터도 딕셔너리로 반환
        metrics = {'symbol': symbol, 'roi': roi, 'win_rate': win_rate, 'mdd': mdd}
        return summary, metrics

    except Exception as e:
        return f"❌ 백테스트 오류: {e}", None

def run_batch_backtest(days=365):
    """일괄 백테스트: 텍스트 리포트만 생성"""
    print(f"🚀 일괄 백테스팅 시작 ({len(CONFIG['SYMBOLS'])}개 심볼)")
    
    lines = [f"📊 **포트폴리오 전체 시뮬레이션 ({days}일)**", "━━━━━━━━━━━━━━━━━━━━"]
    total_roi = 0
    total_win = 0
    count = 0

    for symbol in CONFIG['SYMBOLS']:
        text, metrics = run_single_backtest(symbol, days)
        if metrics:
            icon = "🔴" if metrics['roi'] < 0 else "🟢"
            lines.append(f"{icon} **{symbol}**: {metrics['roi']:+.2f}% (MDD {metrics['mdd']:.1f}%)")
            total_roi += metrics['roi']
            total_win += metrics['win_rate']
            count += 1
        else:
            lines.append(f"⚠️ {symbol}: 실패")

    if count > 0:
        avg_roi = total_roi / count
        avg_win = total_win / count
        header = (
            f"🏆 평균 수익률: **{avg_roi:+.2f}%**\n"
            f"🎯 평균 승률: **{avg_win:.1f}%**\n"
            f"━━━━━━━━━━━━━━━━━━━━"
        )
        lines.insert(2, header)
    
    return "\n".join(lines)

    