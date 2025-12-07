import sys
import os
import pandas as pd

# 1. 현재 파일(main.py)의 위치를 기준으로 상위 폴더(src)의 상위 폴더(deep_stock)를 찾음
current_dir = os.path.dirname(os.path.abspath(__file__)) # src 폴더
root_dir = os.path.dirname(current_dir)                  # deep_stock 폴더

# 2. 프로젝트 루트 경로를 sys.path에 추가
sys.path.append(root_dir)

from config.settings import Config
from src.data_loader import DataLoader
from src.backtester import Backtester
# [중요] Trailing Strategy 사용
from src.strategies.volatility_breakout_trailing import VolatilityBreakoutTrailingStrategy
from src.risk_manager import RiskManager

def run_portfolio_backtest():
    """
    설정된 포트폴리오(BTC, ETH 등)를 순차적으로 백테스트하고 통합 결과를 출력합니다.
    """
    print(f"\n{'='*60}")
    print(f"🚀 AI Crypto Trader - Portfolio Mode")
    print(f"{'='*60}")

    total_initial_balance = 0
    total_final_balance = 0
    portfolio_results = []

    # 1. 포트폴리오 루프 (BTC -> ETH -> ...)
    for symbol, params in Config.PORTFOLIO.items():
        print(f"\nTesting {symbol}...")
        
        # [핵심] Config 전역 설정 덮어씌우기 (이 종목에 맞는 설정 주입)
        Config.SYMBOL = symbol
        Config.SMA_PERIOD = params['SMA_PERIOD']
        Config.LEVERAGE = params['LEVERAGE']
        Config.TRAILING_STOP_MULTIPLIER = params['TRAILING_MULT']
        
        # 자금 배분 (총 자금 100불 가정 시 비중대로 나눔)
        # 예: 100불 * 0.6 = 60불 시작
        allocated_balance = Config.INITIAL_BALANCE * params['ALLOCATION']
        Config.RISK_PER_TRADE = params.get('RISK_PER_TRADE', Config.RISK_PER_TRADE)
        
        # 2. 데이터 로드
        loader = DataLoader()
        # force_update=False로 해서 기존 CSV 활용 (없으면 다운로드)
        df = loader.get_backtest_data(force_update=False)
        
        # 3. 객체 생성
        strategy = VolatilityBreakoutTrailingStrategy()
        
        # 리스크 매니저는 '할당된 자금'만 본다고 가정하고 초기화되진 않지만,
        # 백테스터에 초기 자금을 주입해서 시뮬레이션 함
        risk_manager = RiskManager() 
        
        # 4. 백테스터 실행
        tester = Backtester(df, strategy, risk_manager)
        
        # [중요] 백테스터의 시작 자금을 '할당된 금액'으로 강제 설정
        tester.balance = allocated_balance
        
        # 실행
        report = tester.run()
        
        # 5. 결과 집계
        if report: # 거래가 있었을 경우
            final_bal = tester.balance
            roi = ((final_bal - allocated_balance) / allocated_balance) * 100
            
            portfolio_results.append({
                "Symbol": symbol,
                "Allocated": allocated_balance,
                "Final": final_bal,
                "ROI": roi,
                "MDD": report['mdd_pct']
            })
            
            total_initial_balance += allocated_balance
            total_final_balance += final_bal
        else:
            # 거래 없었음
            total_initial_balance += allocated_balance
            total_final_balance += allocated_balance

    # 6. 최종 포트폴리오 리포트
    print(f"\n\n{'='*60}")
    print(f"🏆 PORTFOLIO FINAL REPORT")
    print(f"{'='*60}")
    
    # 종목별 성과 출력
    df_res = pd.DataFrame(portfolio_results)
    if not df_res.empty:
        print(df_res.to_string(index=False, float_format="%.2f"))
    
    print("-" * 60)
    
    # 통합 성과
    total_roi = ((total_final_balance - total_initial_balance) / total_initial_balance) * 100
    print(f"Total Initial: ${total_initial_balance:.2f}")
    print(f"Total Final:   ${total_final_balance:.2f}")
    print(f"Total ROI:     {total_roi:.2f}%")
    print(f"{'='*60}")

if __name__ == "__main__":
    try:
        run_portfolio_backtest()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()