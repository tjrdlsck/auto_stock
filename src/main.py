import sys
import os
import pandas as pd

# 1. 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__)) 
root_dir = os.path.dirname(current_dir)                  
sys.path.append(root_dir)

from config.settings import Config
from src.data_loader import DataLoader
from src.backtester import Backtester
from src.strategies.volatility_breakout_trailing import VolatilityBreakoutTrailingStrategy
from src.risk_manager import RiskManager

def run_portfolio_backtest():
    """
    [Step 3 Modified]
    설정된 포트폴리오를 순차적으로 실행하며, 종목별 맞춤 파라미터를 주입합니다.
    """
    print(f"\n{'='*60}")
    print(f"🚀 AI Crypto Trader - Portfolio Mode (1h Data)")
    print(f"{'='*60}")

    total_initial_balance = 0
    total_final_balance = 0
    portfolio_results = []
    
    # 포트폴리오 루프
    for symbol, params in Config.PORTFOLIO.items():
        print(f"\n>> Preparing {symbol}...")

        # 1. [Safety Check] 자금 할당이 0이면 스킵 (오류 방지)
        allocation_ratio = params.get('ALLOCATION', 0.0)
        if allocation_ratio <= 0:
            print(f"⚠️ Allocation is 0.0. Skipping {symbol}.")
            continue

        # 2. Config 전역 설정 업데이트 (데이터 로더 등을 위해 필요)
        Config.SYMBOL = symbol
        # 나머지 파라미터는 객체 생성 시 직접 주입하므로 Config를 덮어쓸 필요성이 줄었지만,
        # 로깅이나 기본값 참조를 위해 업데이트해두는 것이 안전함.
        Config.SMA_PERIOD = params.get('SMA_PERIOD', Config.SMA_PERIOD)
        Config.LEVERAGE = params.get('LEVERAGE', Config.LEVERAGE)
        
        # 3. 자금 설정
        allocated_balance = Config.INITIAL_BALANCE * allocation_ratio
        
        # 4. 데이터 로드
        loader = DataLoader()
        # force_update=False: 기존 다운로드된 csv 활용
        df = loader.get_backtest_data(force_update=False)
        
        if df.empty:
            print(f"❌ No data found for {symbol}. Skipping.")
            continue
        
        # 5. [핵심] 전략 파라미터 매핑 및 객체 생성
        # settings.py의 키 이름과 Strategy 클래스가 받는 키 이름을 매칭시켜줍니다.
        strategy_config = {
            'sma_period': params.get('SMA_PERIOD', Config.SMA_PERIOD),
            'k_window': params.get('VBO_K_WINDOW', Config.VBO_K_WINDOW),
            'atr_period': Config.ATR_PERIOD, # ATR 기간은 보통 고정
            
            # [BTC/ETH 튜닝 포인트]
            'sl_multiplier': params.get('SL_MULT', Config.STOP_LOSS_MULTIPLIER),
            'trailing_mult': params.get('TRAILING_MULT', Config.TRAILING_STOP_MULTIPLIER)
        }
        
        strategy = VolatilityBreakoutTrailingStrategy(config=strategy_config)
        
        # 6. 리스크 매니저 설정 주입
        risk_manager = RiskManager(
            risk_per_trade=params.get('RISK_PER_TRADE', Config.RISK_PER_TRADE),
            leverage=params.get('LEVERAGE', Config.LEVERAGE)
        )
        
        # 7. 백테스터 실행
        # symbol 인자를 넘겨주어 로그/파일 저장 시 구분되게 함
        tester = Backtester(df, strategy, risk_manager, symbol=symbol)
        tester.balance = allocated_balance # 할당된 자금 주입
        
        print(f"   Running Backtest... (Allocated: ${allocated_balance:,.2f})")
        report = tester.run()
        
        # 8. 결과 집계
        if report: 
            final_bal = tester.balance
            roi = ((final_bal - allocated_balance) / allocated_balance) * 100
            
            # 포트폴리오 결과 리스트에 추가
            portfolio_results.append({
                "Symbol": symbol,
                "Init_Bal": allocated_balance,
                "Final_Bal": final_bal,
                "ROI(%)": roi,
                "MDD(%)": report['mdd_pct'],
                "WinRate(%)": report['win_rate_pct'],
                "Trades": report['total_trades'],
                "Liq": report['liquidations'],
                "Safety(%)": report['min_safety_margin']
            })
            
            total_initial_balance += allocated_balance
            total_final_balance += final_bal
        else:
            print(f"⚠️ No trades executed for {symbol}.")
            total_initial_balance += allocated_balance
            total_final_balance += allocated_balance

    # 9. 최종 종합 리포트 출력
    print(f"\n\n{'='*80}")
    print(f"🏆 PORTFOLIO FINAL REPORT")
    print(f"{'='*80}")
    
    if portfolio_results:
        df_res = pd.DataFrame(portfolio_results)
        # 보기 좋게 컬럼 순서 정렬 및 포맷팅
        cols = ["Symbol", "Init_Bal", "Final_Bal", "ROI(%)", "MDD(%)", "WinRate(%)", "Trades", "Liq", "Safety(%)"]
        print(df_res[cols].to_string(index=False, float_format="%.2f"))
        
        print("-" * 80)
        
        total_profit = total_final_balance - total_initial_balance
        total_roi = (total_profit / total_initial_balance) * 100 if total_initial_balance > 0 else 0
        
        print(f"TOTAL INITIAL: ${total_initial_balance:,.2f}")
        print(f"TOTAL FINAL:   ${total_final_balance:,.2f}")
        print(f"NET PROFIT:    ${total_profit:,.2f}")
        print(f"TOTAL ROI:     {total_roi:.2f}%")
    else:
        print("No simulation results available.")
        
    print(f"{'='*80}")

if __name__ == "__main__":
    try:
        run_portfolio_backtest()
    except KeyboardInterrupt:
        print("\n🛑 Aborted by user.")
    except Exception as e:
        print(f"\n❌ Critical Error: {e}")
        import traceback
        traceback.print_exc()