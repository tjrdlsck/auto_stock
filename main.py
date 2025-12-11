import sys
import os
import pandas as pd
from config.settings import Config
from src.data_loader import DataLoader
from lib.engine import Engine
from strategies.vbo_strategy import VolatilityBreakoutStrategy

def main():
    print(f"\n{'='*60}")
    print(f"🚀 AI Crypto Trader - Event Driven Backtest (Portfolio Mode)")
    print(f"{'='*60}")

    # 포트폴리오 설정 확인
    targets = list(Config.PORTFOLIO.keys())
    if not targets:
        print("❌ No symbols found in Config.PORTFOLIO.")
        return

    print(f"📋 Target Symbols: {targets}")
    
    # 전체 결과 요약용 리스트
    summary_report = []

    # ---------------------------------------------------------
    # [Loop] 포트폴리오 내 모든 종목에 대해 순차 실행
    # ---------------------------------------------------------
    for symbol in targets:
        print(f"\n{'='*40}")
        print(f"▶️ Processing: {symbol}")
        print(f"{'='*40}")

        # 1. 데이터 로드
        # DataLoader는 Config.SYMBOL을 참조하므로, 루프마다 변경해줘야 함
        Config.SYMBOL = symbol 
        loader = DataLoader()
        
        # force_update=False: 기존 캐시(parquet)가 있으면 사용
        df = loader.get_backtest_data(force_update=False)

        if df.empty:
            print(f"⚠️ Data missing for {symbol}. Skipping...")
            continue

        # 2. 엔진 초기화
        engine = Engine(initial_cash=Config.INITIAL_BALANCE)
        
        # 테스트 기간 설정 (최근 N일)
        test_days = Config.TEST_DAYS
        if len(df) > test_days * 24:
            start_idx = len(df) - (test_days * 24)
            test_df = df.iloc[start_idx:]
        else:
            test_df = df
            
        engine.add_data(test_df)
        print(f"✅ Data Loaded: {len(test_df)} bars")

        # 3. 전략 설정 (Step 2에서 만든 자동 설정 로직 사용)
        strategy_params = Config.get_strategy_config(symbol)
        print(f"⚙️ Strategy Params: {strategy_params}")
        
        engine.add_strategy(VolatilityBreakoutStrategy, **strategy_params)

        # 4. 시뮬레이션 실행
        result = engine.run()

        # [추가] 지표가 포함된 데이터 저장 (Visualizer용)
        # 전략 내부의 df는 이미 indicator가 계산되어 있음
        if hasattr(engine, 'strategy') and hasattr(engine.strategy, 'df'):
            analyzed_df = engine.strategy.df.copy()
            # 파일명: analyzed_BTC_USDT.parquet
            analyzed_path = os.path.join(Config.SAVE_DIR, f"analyzed_{clean_symbol}.parquet")
            analyzed_df.to_parquet(analyzed_path)
            print(f"📊 Analyzed data saved: {analyzed_path}")

        # 5. 결과 파일 저장 (Visualizer 호환용)
        clean_symbol = symbol.replace("/", "_")
        
        # (A) 자산 곡선 (Equity Curve) 저장
        # Step 1에서 Broker에 추가한 history 사용
        if hasattr(engine.broker, 'equity_history') and engine.broker.equity_history:
            df_equity = pd.DataFrame(engine.broker.equity_history)
            equity_path = os.path.join(Config.SAVE_DIR, f"equity_curve_{clean_symbol}.csv")
            df_equity.to_csv(equity_path, index=False)
            print(f"💾 Equity curve saved: {equity_path}")
        
        # (B) 거래 내역 (Trade Log) 저장
        # 파일명 규칙 변경: result_ -> trade_log_
        if result['trades']:
            trades_df = pd.DataFrame([t.__dict__ for t in result['trades']])
            # Enum 객체 문자열 변환
            trades_df['side'] = trades_df['side'].apply(lambda x: x.value)
            
            trade_path = os.path.join(Config.SAVE_DIR, f"trade_log_{clean_symbol}.csv")
            trades_df.to_csv(trade_path, index=False)
            print(f"💾 Trade log saved: {trade_path}")
        else:
            print("ℹ️ No trades executed.")

        # 6. 결과 집계
        summary_report.append({
            'Symbol': symbol,
            'ROI (%)': result['roi_pct'],
            'Win Rate (%)': result['win_rate_pct'],
            'Trades': len(result['trades']),
            'Final Balance': result['final_balance']
        })

    # ---------------------------------------------------------
    # [Final] 전체 포트폴리오 요약 출력
    # ---------------------------------------------------------
    if summary_report:
        print(f"\n\n{'='*60}")
        print(f"🏆 Portfolio Backtest Summary")
        print(f"{'='*60}")
        df_summary = pd.DataFrame(summary_report)
        # 보기 좋게 출력
        print(df_summary.to_string(index=False, float_format="%.2f"))
        
        total_pnl = df_summary['Final Balance'].sum() - (Config.INITIAL_BALANCE * len(df_summary))
        print(f"{'-'*60}")
        print(f"💰 Total Portfolio PnL: ${total_pnl:,.2f}")
        print(f"👉 Now run 'visualize.py' to see the charts.")
        print(f"{'='*60}")

if __name__ == "__main__":
    main()