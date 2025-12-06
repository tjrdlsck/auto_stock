import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# 상위 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from config.settings import Config

def plot_results():
    print("🎨 [Visualization] Generating interactive chart...")

    # 1. 데이터 로드
    try:
        # 가격 데이터
        price_file = os.path.join(Config.SAVE_DIR, f"{Config.SYMBOL.replace('/', '_')}_{Config.TIMEFRAME}.csv")
        df_price = pd.read_csv(price_file, index_col='datetime', parse_dates=True)
        
        # 거래 로그
        trade_file = os.path.join(Config.SAVE_DIR, "trade_log.csv")
        df_trades = pd.read_csv(trade_file, parse_dates=['entry_time', 'exit_time'])
        
        # 자산 곡선 (equity_curve.csv가 없다면 trade_log로 추정 가능하지만, 있는 게 좋음)
        equity_file = os.path.join(Config.SAVE_DIR, "equity_curve.csv")
        df_equity = pd.read_csv(equity_file, parse_dates=['timestamp'])
        df_equity.set_index('timestamp', inplace=True)

    except FileNotFoundError as e:
        print(f"❌ Error: Missing data files. Please run backtest first.\n{e}")
        return

    # 2. 백테스트 기간에 맞춰 차트 범위 자르기
    # 거래가 시작된 시점부터 끝난 시점까지만 차트를 그림 (데이터가 너무 길면 보기가 힘듦)
    if not df_trades.empty:
        start_date = df_trades['entry_time'].min() - pd.Timedelta(days=5)
        end_date = df_trades['exit_time'].max() + pd.Timedelta(days=5)
        df_price = df_price.loc[start_date:end_date]
        df_equity = df_equity.loc[start_date:end_date]
    
    # 3. 서브플롯 생성 (위: 캔들차트+매매타점, 아래: 자산곡선+MDD)
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.05, 
        row_heights=[0.7, 0.3],
        subplot_titles=(f"{Config.SYMBOL} Price & Trades", "Account Balance (Equity Curve)")
    )

    # --- [상단] 캔들 차트 ---
    fig.add_trace(go.Candlestick(
        x=df_price.index,
        open=df_price['open'], high=df_price['high'],
        low=df_price['low'], close=df_price['close'],
        name='Price'
    ), row=1, col=1)

    # --- [상단] 매매 타점 표시 ---
    # Buy Entry (초록색 삼각형 ▲)
    buys = df_trades[df_trades['type'] == 'BUY']
    fig.add_trace(go.Scatter(
        x=buys['entry_time'], y=buys['entry_price'],
        mode='markers', marker=dict(symbol='triangle-up', color='lime', size=12),
        name='Buy Entry', hovertext=buys['type']
    ), row=1, col=1)

    # Sell Entry (빨간색 삼각형 ▼)
    sells = df_trades[df_trades['type'] == 'SELL']
    fig.add_trace(go.Scatter(
        x=sells['entry_time'], y=sells['entry_price'],
        mode='markers', marker=dict(symbol='triangle-down', color='red', size=12),
        name='Sell Entry', hovertext=sells['type']
    ), row=1, col=1)

    # Exit Points (수익: 파란 원, 손실: 주황 원)
    # 청산 당한 경우(해골) 별도 표시
    wins = df_trades[df_trades['pnl'] > 0]
    losses = df_trades[(df_trades['pnl'] <= 0) & (~df_trades['reason'].str.contains("LIQUIDATION"))]
    liquidations = df_trades[df_trades['reason'].str.contains("LIQUIDATION")]

    # 익절 (파란 원) - 정확한 exit_price 사용
    fig.add_trace(go.Scatter(
        x=wins['exit_time'], y=wins['exit_price'], 
        mode='markers', marker=dict(symbol='circle', color='cyan', size=8),
        name='Win Exit', 
        hovertext=wins.apply(lambda x: f"PnL: ${x['pnl']:.2f}<br>{x['reason']}", axis=1)
    ), row=1, col=1)

    # 손절 (주황 원) - 정확한 exit_price 사용
    fig.add_trace(go.Scatter(
        x=losses['exit_time'], y=losses['exit_price'],
        mode='markers', marker=dict(symbol='circle', color='orange', size=8),
        name='Loss Exit',
        hovertext=losses.apply(lambda x: f"PnL: ${x['pnl']:.2f}<br>{x['reason']}", axis=1)
    ), row=1, col=1)
    
    # 청산 (검정/해골 표시 X)
    if not liquidations.empty:
        fig.add_trace(go.Scatter(
            x=liquidations['exit_time'], y=liquidations['liq_price'],
            mode='markers', marker=dict(symbol='x', color='black', size=15),
            name='LIQUIDATED',
            hovertext="☠️ LIQUIDATION"
        ), row=1, col=1)


    # --- [하단] 자산 곡선 (Equity Curve) ---
    fig.add_trace(go.Scatter(
        x=df_equity.index, y=df_equity['equity'],
        mode='lines', line=dict(color='yellow', width=2),
        name='Balance', fill='tozeroy' # 아래를 채워서 더 보기 좋게
    ), row=2, col=1)

    # 레이아웃 꾸미기 (다크 모드)
    fig.update_layout(
        template='plotly_dark',
        title=f"Backtest Analysis: {Config.SYMBOL}",
        xaxis_rangeslider_visible=False,
        height=800
    )

    # 4. 파일 저장 및 실행
    output_file = "backtest_result.html"
    fig.write_html(output_file)
    print(f"✨ Chart saved to {output_file}")
    print(f"👉 Please open '{output_file}' in your web browser.")

if __name__ == "__main__":
    plot_results()