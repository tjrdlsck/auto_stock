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
    data 폴더를 스캔하여 백테스트 결과(trade_log)가 존재하는 심볼 목록을 반환합니다.
    파일명 패턴: trade_log_{SYMBOL}.csv
    """
    # trade_log_로 시작하는 CSV 파일 검색
    pattern = os.path.join(Config.SAVE_DIR, "trade_log_*.csv")
    files = glob.glob(pattern)
    
    symbols = []
    for f in files:
        # 파일명에서 심볼 추출 
        # 예: .../trade_log_BTC_USDT.csv -> BTC_USDT
        filename = os.path.basename(f)
        if filename.startswith("trade_log_") and filename.endswith(".csv"):
            symbol = filename[10:-4] # "trade_log_"(10글자) 제거, ".csv"(4글자) 제거
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
        # (A) Price Data 로드 (Parquet 우선 지원)
        price_parquet = os.path.join(Config.SAVE_DIR, f"{target_symbol}_{Config.TIMEFRAME}.parquet")
        price_csv = os.path.join(Config.SAVE_DIR, f"{target_symbol}_{Config.TIMEFRAME}.csv")
        
        df_price = pd.DataFrame()
        
        # Parquet이 있으면 로드, 없으면 CSV 확인
        if os.path.exists(price_parquet):
            df_price = pd.read_parquet(price_parquet)
            print(f"✅ Loaded Price Data (Parquet): {len(df_price)} bars")
        elif os.path.exists(price_csv):
            df_price = pd.read_csv(price_csv, index_col='datetime', parse_dates=True)
            print(f"✅ Loaded Price Data (CSV): {len(df_price)} bars")
        else:
            print(f"⚠️ Warning: Price file not found. Chart might be incomplete.")

        # (B) Trade Log 로드
        trade_file = os.path.join(Config.SAVE_DIR, f"trade_log_{target_symbol}.csv")
        df_trades = pd.read_csv(trade_file, parse_dates=['entry_time', 'exit_time'])
        
        # [수정] 컬럼 호환성 처리 ('type' -> 'side')
        if 'side' not in df_trades.columns and 'type' in df_trades.columns:
            df_trades.rename(columns={'type': 'side'}, inplace=True)
            
        print(f"✅ Loaded Trade Log: {len(df_trades)} trades")

        # (C) Equity Curve 로드
        equity_file = os.path.join(Config.SAVE_DIR, f"equity_curve_{target_symbol}.csv")
        if os.path.exists(equity_file):
            df_equity = pd.read_csv(equity_file, parse_dates=['timestamp'])
            df_equity.set_index('timestamp', inplace=True)
            print(f"✅ Loaded Equity Curve: {len(df_equity)} records")
        else:
            df_equity = pd.DataFrame()
            print("ℹ️ Equity curve file not found. Skipping balance chart.")

    except Exception as e:
        print(f"❌ Error loading data: {e}")
        # 디버깅을 위해 상세 에러 출력
        import traceback
        traceback.print_exc()
        return

    # 4. 차트 범위 최적화 (거래 기간만 확대)
    if not df_trades.empty and not df_price.empty:
        # 첫 거래 2일 전 ~ 마지막 거래 2일 후로 범위 설정
        start_date = df_trades['entry_time'].min() - pd.Timedelta(hours=48)
        end_date = df_trades['exit_time'].max() + pd.Timedelta(hours=48)
        
        df_price = df_price.loc[start_date:end_date]
        if not df_equity.empty:
            df_equity = df_equity.loc[start_date:end_date]

    # 5. 서브플롯 생성
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.05, 
        row_heights=[0.7, 0.3],
        subplot_titles=(f"{target_symbol} Price & Trades ({Config.TIMEFRAME})", "Account Balance")
    )

    # --- [상단] 캔들 차트 ---
    if not df_price.empty:
        fig.add_trace(go.Candlestick(
            x=df_price.index,
            open=df_price['open'], high=df_price['high'],
            low=df_price['low'], close=df_price['close'],
            name='Price'
        ), row=1, col=1)

    # --- [상단] 매매 타점 (FIX: 'type' -> 'side') ---
    # Buy Entry
    if not df_trades.empty:
        # 'side' 컬럼 사용
        buys = df_trades[df_trades['side'] == 'BUY']
        if not buys.empty:
            fig.add_trace(go.Scatter(
                x=buys['entry_time'], y=buys['entry_price'],
                mode='markers', marker=dict(symbol='triangle-up', color='lime', size=12),
                name='Buy Entry'
            ), row=1, col=1)

        # Sell Entry
        sells = df_trades[df_trades['side'] == 'SELL']
        if not sells.empty:
            fig.add_trace(go.Scatter(
                x=sells['entry_time'], y=sells['entry_price'],
                mode='markers', marker=dict(symbol='triangle-down', color='red', size=12),
                name='Sell Entry'
            ), row=1, col=1)

        # Win Exits
        wins = df_trades[df_trades['net_pnl'] > 0]
        if not wins.empty:
            fig.add_trace(go.Scatter(
                x=wins['exit_time'], y=wins['exit_price'], 
                mode='markers', marker=dict(symbol='circle', color='cyan', size=8),
                name='Win Exit',
                hovertext=wins.apply(lambda x: f"PnL: ${x['net_pnl']:.2f}<br>{x['exit_reason']}", axis=1)
            ), row=1, col=1)

        # Loss Exits (LIQUIDATION 제외)
        losses = df_trades[(df_trades['net_pnl'] <= 0) & (df_trades['exit_reason'] != "LIQUIDATION")]
        if not losses.empty:
            fig.add_trace(go.Scatter(
                x=losses['exit_time'], y=losses['exit_price'],
                mode='markers', marker=dict(symbol='circle', color='orange', size=8),
                name='Loss Exit',
                hovertext=losses.apply(lambda x: f"PnL: ${x['net_pnl']:.2f}<br>{x['exit_reason']}", axis=1)
            ), row=1, col=1)
            
        # Liquidations
        liqs = df_trades[df_trades['exit_reason'] == "LIQUIDATION"]
        if not liqs.empty:
            fig.add_trace(go.Scatter(
                x=liqs['exit_time'], y=liqs['exit_price'],
                mode='markers', marker=dict(symbol='x', color='white', size=15, line=dict(width=2, color='red')),
                name='LIQUIDATION',
                hovertext="☠️ LIQUIDATION"
            ), row=1, col=1)

    # --- [하단] 자산 곡선 ---
    if not df_equity.empty:
        fig.add_trace(go.Scatter(
            x=df_equity.index, y=df_equity['equity'],
            mode='lines', line=dict(color='yellow', width=2),
            name='Balance', fill='tozeroy'
        ), row=2, col=1)

    # 레이아웃 설정
    fig.update_layout(
        template='plotly_dark',
        title=f"Backtest Result: {target_symbol}",
        xaxis_rangeslider_visible=False,
        height=900
    )

    # 저장
    output_file = f"backtest_result_{target_symbol}.html"
    fig.write_html(output_file)
    print(f"\n✨ Chart saved to {output_file}")
    print(f"👉 Open this file in your browser to view the interactive chart.")

if __name__ == "__main__":
    plot_results()