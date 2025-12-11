import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
import glob

# 상위 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from config.settings import Config

def get_available_symbols():
    """
    data 폴더를 스캔하여 백테스트 결과가 존재하는 심볼 목록을 반환합니다.
    파일명 패턴: trade_log_{SYMBOL}.csv
    """
    pattern = os.path.join(Config.SAVE_DIR, "trade_log_*.csv")
    files = glob.glob(pattern)
    
    symbols = []
    for f in files:
        # 파일명에서 심볼 추출 (예: trade_log_BTC_USDT.csv -> BTC_USDT)
        filename = os.path.basename(f)
        symbol = filename.replace("trade_log_", "").replace(".csv", "")
        symbols.append(symbol)
    
    return sorted(symbols)

def plot_results():
    print(f"\n{'='*60}")
    print("🎨 AI Crypto Trader - Visualizer")
    print(f"{'='*60}")

    # 1. 분석 가능한 종목 찾기
    symbols = get_available_symbols()
    
    if not symbols:
        print(f"❌ No backtest results found in '{Config.SAVE_DIR}'.")
        print("👉 Please run 'main.py' first.")
        return

    # 2. 사용자 선택 (Interactive)
    print("🔍 Found results for:")
    for idx, sym in enumerate(symbols):
        print(f"  {idx + 1}. {sym}")
    
    try:
        selection = input(f"\n👉 Select symbol number (1-{len(symbols)}, default 1): ").strip()
        if not selection:
            choice = 0
        else:
            choice = int(selection) - 1
            if choice < 0 or choice >= len(symbols):
                print("⚠️ Invalid number. Using default (1).")
                choice = 0
    except ValueError:
        print("⚠️ Invalid input. Using default (1).")
        choice = 0

    target_symbol = symbols[choice]
    print(f"\n🚀 Visualizing: {target_symbol} ...")

    # 3. 데이터 로드
    try:
        # 파일명 규칙 재구성
        # Price Data: {SYMBOL}_{TIMEFRAME}.csv (예: BTC_USDT_1h.csv)
        price_file = os.path.join(Config.SAVE_DIR, f"{target_symbol}_{Config.TIMEFRAME}.csv")
        
        # Result Data: trade_log_{SYMBOL}.csv
        trade_file = os.path.join(Config.SAVE_DIR, f"trade_log_{target_symbol}.csv")
        equity_file = os.path.join(Config.SAVE_DIR, f"equity_curve_{target_symbol}.csv")

        # 로드
        if not os.path.exists(price_file):
            print(f"⚠️ Warning: Price file not found ({price_file}). Chart might be incomplete.")
            df_price = pd.DataFrame()
        else:
            df_price = pd.read_csv(price_file, index_col='datetime', parse_dates=True)

        df_trades = pd.read_csv(trade_file, parse_dates=['entry_time', 'exit_time'])
        
        df_equity = pd.read_csv(equity_file, parse_dates=['timestamp'])
        df_equity.set_index('timestamp', inplace=True)

    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    # 4. 차트 범위 최적화 (거래 기간만 확대)
    if not df_trades.empty and not df_price.empty:
        start_date = df_trades['entry_time'].min() - pd.Timedelta(hours=12) # 1시간봉이므로 여유 12시간
        end_date = df_trades['exit_time'].max() + pd.Timedelta(hours=12)
        
        # 데이터가 너무 크면 Plotly가 느려지므로 범위 자르기
        df_price = df_price.loc[start_date:end_date]
        df_equity = df_equity.loc[start_date:end_date]

    # 5. 서브플롯 생성
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.05, 
        row_heights=[0.7, 0.3],
        subplot_titles=(f"{target_symbol} Price & Trades ({Config.TIMEFRAME})", "Account Balance (Isolated)")
    )

    # --- [상단] 캔들 차트 ---
    if not df_price.empty:
        fig.add_trace(go.Candlestick(
            x=df_price.index,
            open=df_price['open'], high=df_price['high'],
            low=df_price['low'], close=df_price['close'],
            name='Price'
        ), row=1, col=1)

    # --- [상단] 매매 타점 ---
    # Buy Entry
    buys = df_trades[df_trades['type'] == 'BUY']
    if not buys.empty:
        fig.add_trace(go.Scatter(
            x=buys['entry_time'], y=buys['entry_price'],
            mode='markers', marker=dict(symbol='triangle-up', color='lime', size=12),
            name='Buy Entry', hovertext=buys['type']
        ), row=1, col=1)

    # Sell Entry
    sells = df_trades[df_trades['type'] == 'SELL']
    if not sells.empty:
        fig.add_trace(go.Scatter(
            x=sells['entry_time'], y=sells['entry_price'],
            mode='markers', marker=dict(symbol='triangle-down', color='red', size=12),
            name='Sell Entry', hovertext=sells['type']
        ), row=1, col=1)

    # Exits (Win/Loss/Liq)
    wins = df_trades[df_trades['pnl'] > 0]
    losses = df_trades[(df_trades['pnl'] <= 0) & (~df_trades['reason'].str.contains("LIQUIDATION"))]
    liquidations = df_trades[df_trades['reason'].str.contains("LIQUIDATION")]

    # Win
    if not wins.empty:
        fig.add_trace(go.Scatter(
            x=wins['exit_time'], y=wins['exit_price'], 
            mode='markers', marker=dict(symbol='circle', color='cyan', size=8),
            name='Win Exit', 
            hovertext=wins.apply(lambda x: f"PnL: ${x['pnl']:.2f}<br>{x['reason']}", axis=1)
        ), row=1, col=1)

    # Loss
    if not losses.empty:
        fig.add_trace(go.Scatter(
            x=losses['exit_time'], y=losses['exit_price'],
            mode='markers', marker=dict(symbol='circle', color='orange', size=8),
            name='Loss Exit',
            hovertext=losses.apply(lambda x: f"PnL: ${x['pnl']:.2f}<br>{x['reason']}", axis=1)
        ), row=1, col=1)
    
    # Liquidation (해골)
    if not liquidations.empty:
        fig.add_trace(go.Scatter(
            x=liquidations['exit_time'], y=liquidations['liq_price'], # 청산은 exit_price 대신 liq_price 표시가 더 정확할 수 있음
            mode='markers', marker=dict(symbol='x', color='white', size=12, line=dict(width=2, color='red')),
            name='LIQUIDATED',
            hovertext="☠️ LIQUIDATION"
        ), row=1, col=1)

    # --- [하단] 자산 곡선 ---
    if not df_equity.empty:
        fig.add_trace(go.Scatter(
            x=df_equity.index, y=df_equity['equity'],
            mode='lines', line=dict(color='yellow', width=2),
            name='Balance', fill='tozeroy'
        ), row=2, col=1)

    # 레이아웃
    fig.update_layout(
        template='plotly_dark',
        title=f"Backtest Result: {target_symbol}",
        xaxis_rangeslider_visible=False,
        height=900
    )

    # 저장
    output_file = f"backtest_result_{target_symbol}.html"
    fig.write_html(output_file)
    print(f"✨ Chart saved to {output_file}")
    print(f"👉 Please open '{output_file}' in your web browser.")

if __name__ == "__main__":
    plot_results()