import sys
import os
import matplotlib.pyplot as plt
import tempfile
import backtrader as bt
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

# ----------------------------------------------------------- 
# [Linux Server Config]
# 서버 환경(No Monitor)에서 차트 생성 시 에러 방지
# ----------------------------------------------------------- 
plt.switch_backend('Agg') 

# 프로젝트 루트 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import CONFIG, DATA_DIR
from alpha_layer.brain import Brain

# --------------------------------------------------------- 
# 1. 데이터 피드 클래스
# --------------------------------------------------------- 
class HMMData(bt.feeds.PandasData):
    lines = ('regime',)
    params = (('regime', -1),)

# --------------------------------------------------------- 
# 2. 커미션 & 레버리지 클래스
# --------------------------------------------------------- 
class FuturesComm(bt.CommInfoBase):
    params = (
        ('stocklike', False), 
        ('commtype', bt.CommInfoBase.COMM_PERC),
        ('perc', 0.0005), 
        ('leverage', 1.0), # 기본 레버리지
        ('slippage_perc', 0.0)
    )

    def _getcommission(self, size, price, pseudoexec=None, **kwargs):
        # 수수료 계산 (레버리지 포함 전체 거래대금 기준)
        commission = super()._getcommission(size, price, pseudoexec, **kwargs)
        # 슬리피지 비용 추가
        slippage_cost = abs(size) * price * self.p.slippage_perc
        return commission + slippage_cost

    def get_margin(self, price): 
        # 선물 거래소 증거금 계산 (가격 / 레버리지)
        return price / self.p.leverage

# --------------------------------------------------------- 
# 3. 전략 클래스 (V13.0 - Unconstrained Trend Following)
# --------------------------------------------------------- 
class HMM_Pro_Strategy_V5(bt.Strategy):
    params = (
        ('bull_id', None), ('bear_id', None),
        ('leverage', 2.0),
        ('atr_period', 14),
        ('rsi_period', 14),
        ('ema_period', 20),
        
        # [진입 조건 완화]
        ('rsi_buy_upper', 65),  # 65 이하면 매수 (기존 55보다 완화)
        ('rsi_buy_lower', 30),  # 과매도 바닥 방지
        ('rsi_sell_lower', 35), # 35 이상이면 매도 (기존 45보다 완화)
        ('rsi_sell_upper', 70), # 과매수 꼭지 방지
        
        # [리스크 관리]
        ('stop_atr', 2.0),      # 손절: 2.0 ATR (변동성 고려 넉넉하게)
        
        # [트레일링 스탑]
        ('trail_trigger', 2.0), # 수익이 2.0 ATR 넘으면 발동
        ('trail_dist', 2.0),    # 고점에서 2.0 ATR 간격 유지하며 추적
        
        ('cooldown_bars', 2),   # 진입/청산 후 쿨타임
        ('debug', False)
    )

    def __init__(self):
        self.regime = self.data.regime
        
        # 지표 정의
        self.atr = bt.indicators.ATR(self.data, period=self.p.atr_period)
        self.rsi = bt.indicators.RSI(self.data, period=self.p.rsi_period)
        self.ema = bt.indicators.EMA(self.data.close, period=self.p.ema_period)

        # 상태 변수
        self.order = None
        self.entry_price = 0.0
        self.stop_price = 0.0
        self.cooldown = 0
        self.trade_log = []

    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                self.entry_price = order.executed.price
                # 진입 시점 ATR로 초기 스탑로스 설정
                self.stop_price = self.entry_price - (self.atr[0] * self.p.stop_atr)
            elif order.issell():
                self.entry_price = order.executed.price
                self.stop_price = self.entry_price + (self.atr[0] * self.p.stop_atr)
            
            self.order = None
            self.cooldown = 0 # 진입 시 쿨타임 초기화
            
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed: return
        
        # 수수료 포함 순수익
        pnl_net = trade.pnlcomm 
        # 진입 가치
        entry_val = abs(trade.size) * trade.price
        # 수익률 (%)
        pnl_pct = (pnl_net / entry_val) * 100 if entry_val > 0 else 0
        
        self.trade_log.append({
            'Entry_Date': bt.num2date(trade.dtopen),
            'Exit_Date': bt.num2date(trade.dtclose),
            'Symbol': self.data._name,
            'Type': 'Long' if trade.long else 'Short',
            'Entry_Price': trade.price,
            'Exit_Price': trade.price + (trade.pnlcomm / trade.size if trade.size!=0 else 0),
            'Size': trade.size,
            'PnL_USDT': pnl_net,
            'PnL_Percent': pnl_pct,
            'Duration_Bars': trade.barlen,
            'Exit_Reason': 'Closed'
        })

    def next(self):
        if self.order: return
        
        close = self.data.close[0]
        low = self.data.low[0]
        high = self.data.high[0]
        regime = int(self.regime[0])
        atr = self.atr[0]
        rsi = self.rsi[0]
        ema = self.ema[0]
        
        if self.cooldown > 0: self.cooldown -= 1

        # ---------------------------------------------
        # 1. 포지션 보유 중 (청산 로직)
        # ---------------------------------------------
        if self.position.size != 0:
            
            # Long Position
            if self.position.size > 0: 
                # Stop Loss Check
                if low < self.stop_price: 
                    self.close()
                    self.cooldown = self.p.cooldown_bars
                    return
                
                # Trailing Stop Update
                # 현재가가 (진입가 + 2ATR) 이상으로 오르면 트레일링 시작
                if close > self.entry_price + (atr * self.p.trail_trigger):
                    new_stop = close - (atr * self.p.trail_dist)
                    # 기존 스탑보다 높을 때만 업데이트 (스탑은 위로만 움직임)
                    if new_stop > self.stop_price:
                        self.stop_price = new_stop
                
                # 국면 전환 시 즉시 청산
                if regime == self.p.bear_id:
                    self.close()
                    self.cooldown = self.p.cooldown_bars
                    return

            # Short Position
            elif self.position.size < 0: 
                # Stop Loss Check
                if high > self.stop_price: 
                    self.close()
                    self.cooldown = self.p.cooldown_bars
                    return
                
                # Trailing Stop Update
                if close < self.entry_price - (atr * self.p.trail_trigger):
                    new_stop = close + (atr * self.p.trail_dist)
                    # 기존 스탑보다 낮을 때만 업데이트 (스탑은 아래로만 움직임)
                    if new_stop < self.stop_price:
                        self.stop_price = new_stop

                # 국면 전환 시 즉시 청산
                if regime == self.p.bull_id:
                    self.close()
                    self.cooldown = self.p.cooldown_bars
                    return

        # ---------------------------------------------
        # 2. 포지션 없음 (진입 로직)
        # ---------------------------------------------
        elif self.position.size == 0 and self.cooldown == 0:
            
            # [레버리지 적용 수량 계산]
            cash = self.broker.get_cash()
            target_value = cash * 0.95 * self.p.leverage
            size = target_value / close

            # Bull Market Entry: EMA 위 + RSI 적정 범위 (눌림목)
            if regime == self.p.bull_id:
                if close > ema and self.p.rsi_buy_lower < rsi < self.p.rsi_buy_upper:
                    self.order = self.buy(size=size)
            
            # Bear Market Entry: EMA 아래 + RSI 적정 범위 (반등)
            elif regime == self.p.bear_id:
                if close < ema and self.p.rsi_sell_lower < rsi < self.p.rsi_sell_upper:
                    self.order = self.sell(size=size)

# --------------------------------------------------------- 
# 4. 실행 함수 (Runner)
# --------------------------------------------------------- 
def run_walk_forward(symbol, total_test_days=180, train_days=365, step_days=30):
    """
    Walk-Forward Analysis (전진 분석) 수행 함수
    """
    brain = Brain()
    df_full, _ = brain.load_data(symbol)

    if df_full is None or len(df_full) < (train_days + total_test_days):
        return f"❌ 데이터 부족 ({symbol})", None, None

    initial_cash = 10000.0
    current_cash = initial_cash
    
    end_date = df_full.index[-1]
    
    # WFA 루프 준비
    steps = []
    remaining_days = total_test_days
    current_end = end_date

    while remaining_days > 0:
        current_step_days = min(step_days, remaining_days)
        test_end = current_end
        test_start = test_end - timedelta(days=current_step_days)
        steps.append((test_start, test_end))
        current_end = test_start
        remaining_days -= current_step_days

    all_trade_logs = []

    # WFA 루프 실행
    for i, (test_start, test_end) in enumerate(steps):
        train_end = test_start - timedelta(seconds=1)
        train_start = train_end - timedelta(days=train_days)

        df_train = df_full.loc[train_start:train_end].copy()
        df_test = df_full.loc[test_start:test_end].copy()

        if len(df_train) < 100 or len(df_test) < 10:
            continue

        # HMM 학습 & 국면 식별
        try:
            model, df_train_res = brain.train_model(df_train, symbol)
            regime_map, stats = brain.identify_regimes(df_train_res)
            sorted_stats = stats.sort_values()
            bear_id = sorted_stats.index[0]
            bull_id = sorted_stats.index[-1]

            # 예측
            test_features = df_test[['Log_Returns_Scaled', 'Range_Vol_Scaled', 'RSI_14_Scaled', 'OBV_Scaled']].values
            df_test['Regime'] = model.predict(test_features)
            
            # Backtrader 실행
            cerebro = bt.Cerebro()
            data_feed = HMMData(dataname=df_test, name=symbol)
            cerebro.adddata(data_feed)

            is_btc = 'BTC' in symbol
            lev = 2.0 if is_btc else 1.0
            
            # 전략 추가 (V13.0 파라미터)
            cerebro.addstrategy(HMM_Pro_Strategy_V5,
                                bull_id=bull_id, bear_id=bear_id,
                                leverage=lev)

            cerebro.broker.setcash(current_cash)
            
            # [중요] 커미션 정보에 레버리지 전달하여 마진 계산 활성화
            cerebro.broker.addcommissioninfo(FuturesComm(
                commission=CONFIG['COMMISSION'], 
                leverage=lev, 
                slippage_perc=CONFIG['SLIPPAGE_PCT']
            ))
            
            # 분석기 추가
            cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
            
            results = cerebro.run()
            strat = results[0]
            current_cash = cerebro.broker.getvalue()
            
            if strat.trade_log:
                all_trade_logs.extend(strat.trade_log)
                
        except Exception as e:
            print(f"⚠️ 구간 {i} 실행 중 오류: {e}")
            continue

    # 최종 결과 계산
    total_roi = (current_cash - initial_cash) / initial_cash * 100
    
    csv_path = None
    if all_trade_logs:
        try:
            df_logs = pd.DataFrame(all_trade_logs)
            cols = ['Symbol', 'Type', 'Entry_Date', 'Exit_Date', 'Entry_Price', 'Exit_Price', 'Size', 'PnL_USDT', 'PnL_Percent', 'Duration_Bars']
            df_logs = df_logs[cols]
            
            clean_symbol = symbol.replace('/', '')
            temp_csv = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", prefix=f"TradeLog_{clean_symbol}_")
            df_logs.to_csv(temp_csv.name, index=False)
            csv_path = temp_csv.name
        except Exception as e:
            print(f"❌ CSV 생성 실패: {e}")

    summary = (
        f"📊 **{symbol} WFA 검증 결과** (최근 {total_test_days}일)\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"💰 최종 수익률: **{total_roi:+.2f}%**\n"
        f"💵 최종 잔고: ${current_cash:,.0f}\n"
        f"📉 총 거래 횟수: {len(all_trade_logs)}회\n"
        f"📄 상세 내역은 첨부된 CSV 파일을 확인하세요."
    )
    
    return summary, {'roi': total_roi}, csv_path


def run_batch_backtest(train_period_days=365, test_period_days=30):
    """일괄 백테스트"""
    print(f"🚀 일괄 백테스팅 시작 ({len(CONFIG['SYMBOLS'])}개 심볼)")
    
    lines = [f"📊 **포트폴리오 전체 시뮬레이션 (WFA Mode)**", "━━━━━━━━━━━━━━━━━━━━"]
    total_roi = 0
    count = 0

    for symbol in CONFIG['SYMBOLS']:
        summary, metrics, _ = run_walk_forward(symbol, total_test_days=test_period_days, train_days=train_period_days)
        if metrics:
            icon = "🔴" if metrics['roi'] < 0 else "🟢"
            lines.append(f"{icon} **{symbol}**: {metrics['roi']:+.2f}%")
            total_roi += metrics['roi']
            count += 1
        else:
            lines.append(f"⚠️ {summary}")

    if count > 0:
        avg_roi = total_roi / count
        header = (
            f"🏆 평균 수익률: **{avg_roi:+.2f}%**\n"
            f"━━━━━━━━━━━━━━━━━━━━"
        )
        lines.insert(2, header)
    
    return "\n".join(lines)

if __name__ == '__main__':
    # 테스트용 실행 코드
    if len(sys.argv) > 1:
        symbol_to_test = sys.argv[1]
        summary, _, path = run_walk_forward(symbol_to_test, total_test_days=180, train_days=365)
        print(summary)
        if path: print(f"Log saved to: {path}")
    else:
        print(run_batch_backtest())