import sys
import os
import pandas as pd
from config.settings import Config
from src.data_loader import DataLoader
from lib.engine import Engine
from strategies.vbo_strategy import VolatilityBreakoutStrategy

def main():
    print(f"\n{'='*60}")
    print(f"🚀 AI Crypto Trader - Event Driven Backtest")
    print(f"{'='*60}")

    # 1. 설정 로드 및 데이터 준비
    # Config에서 정의된 포트폴리오 중 하나를 선택하거나 기본값 사용
    target_symbol = "SOL/USDT" 
    # target_symbol = Config.SYMBOL # settings.py의 기본값 사용 시
    
    print(f"📊 Loading Data for {target_symbol} ({Config.TIMEFRAME})...")
    
    # 기존 DataLoader 재사용 (데이터 페칭 및 캐싱 기능 활용)
    # 주의: DataLoader 내부적으로 Config.SYMBOL을 쓰므로, 필요시 인스턴스 생성 전 Config 수정
    Config.SYMBOL = target_symbol 
    loader = DataLoader()
    df = loader.get_backtest_data(force_update=False)

    if df.empty:
        print("❌ Error: Data not found. Please check connection or symbol.")
        return

    # 2. 엔진 초기화
    initial_balance = 100.0
    engine = Engine(initial_cash=initial_balance)
    
    # 3. 데이터 주입
    # 최근 N일 데이터만 슬라이싱 (테스트 기간 설정)
    test_days = Config.TEST_DAYS
    start_idx = max(0, len(df) - (test_days * 24))
    test_df = df.iloc[start_idx:]
    
    engine.add_data(test_df)
    print(f"✅ Data Ready: {len(test_df)} bars ({test_df.index[0]} ~ {test_df.index[-1]})")

    # 4. 전략 설정 및 등록
    # settings.py의 포트폴리오 설정 가져오기
    portfolio_params = Config.PORTFOLIO.get(target_symbol, {})
    
    # 전략 파라미터 병합 (포트폴리오 설정이 우선, 없으면 기본값)
    strategy_params = {
        'symbol': target_symbol,
        'k_window': portfolio_params.get('VBO_K_WINDOW', Config.VBO_K_WINDOW),
        'sma_period': portfolio_params.get('SMA_PERIOD', Config.SMA_PERIOD),
        'sl_mult': portfolio_params.get('SL_MULT', Config.STOP_LOSS_MULTIPLIER),
        'trailing_mult': portfolio_params.get('TRAILING_MULT', Config.TRAILING_STOP_MULTIPLIER),
        'risk_per_trade': portfolio_params.get('RISK_PER_TRADE', Config.RISK_PER_TRADE),
        'leverage': portfolio_params.get('LEVERAGE', Config.LEVERAGE)
    }

    print(f"⚙️ Strategy Params: {strategy_params}")
    engine.add_strategy(VolatilityBreakoutStrategy, **strategy_params)

    # 5. 시뮬레이션 실행
    print(f"\n▶️ Running Simulation...")
    result = engine.run()

    # 6. 결과 출력
    print(f"\n{'='*60}")
    print(f"🏆 Final Result: {target_symbol}")
    print(f"{'='*60}")
    print(f"💰 Initial Balance: ${result['initial_balance']:,.2f}")
    print(f"💰 Final Balance:   ${result['final_balance']:,.2f}")
    print(f"📈 ROI:             {result['roi_pct']:.2f}%")
    print(f"🎲 Win Rate:        {result['win_rate_pct']:.2f}% ({len(result['trades'])} trades)")
    print(f"{'='*60}")

    # 거래 내역 파일 저장 (선택 사항)
    if result['trades']:
        trades_df = pd.DataFrame([t.__dict__ for t in result['trades']])
        # 객체 내부의 enum이나 datetime 등을 문자열로 변환해야 깔끔하게 저장됨
        trades_df['side'] = trades_df['side'].apply(lambda x: x.value)
        
        save_path = os.path.join(Config.SAVE_DIR, f"result_{target_symbol.replace('/','_')}.csv")
        trades_df.to_csv(save_path, index=False)
        print(f"💾 Trade log saved to: {save_path}")

if __name__ == "__main__":
    main()