import sys
import os
import pandas as pd
import concurrent.futures
from functools import partial

# 1. 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

from config.settings import Config
from src.data_loader import DataLoader
from src.backtester import Backtester
from src.strategies.volatility_breakout_trailing import VolatilityBreakoutTrailingStrategy
from src.risk_manager import RiskManager

def run_single_backtest(task):
    """
    단일 종목 백테스팅 함수 (ProcessPoolExecutor에서 실행됨)
    :param task: (symbol, params, initial_balance) 튜플
    :return: 결과 딕셔너리 or None
    """
    symbol, params, allocated_balance = task
    
    # [주의] 멀티프로세싱 환경에서는 전역 Config가 복제되므로
    # 각 프로세스 내에서 필요한 설정을 다시 세팅하거나 안전하게 사용해야 합니다.
    # 여기서는 파라미터를 직접 객체에 주입하는 방식을 사용하므로 안전합니다.
    
    # Config.SYMBOL을 변경하는 방식은 멀티프로세싱에서 위험할 수 있으므로
    # DataLoader가 인스턴스 생성 시 symbol을 받도록 수정하거나, 
    # 여기서 Config.SYMBOL을 일시적으로 설정합니다 (프로세스 격리 덕분에 안전)
    Config.SYMBOL = symbol 
    
    try:
        loader = DataLoader()
        df = loader.get_backtest_data(force_update=False)
        
        if df.empty:
            return {
                 "Symbol": symbol,
                 "Init_Bal": allocated_balance,
                 "Final_Bal": allocated_balance,
                 "ROI(%)": 0.0,
                 "MDD(%)": 0.0,
                 "WinRate(%)": 0.0,
                 "Trades": 0,
                 "Liq": 0,
                 "Safety(%)": 0.0
            }
            
        # 2. 전략 설정
        strategy_config = {
            'sma_period': params.get('SMA_PERIOD', Config.SMA_PERIOD),
            'k_window': params.get('VBO_K_WINDOW', Config.VBO_K_WINDOW),
            'atr_period': Config.ATR_PERIOD,
            'sl_multiplier': params.get('SL_MULT', Config.STOP_LOSS_MULTIPLIER),
            'trailing_mult': params.get('TRAILING_MULT', Config.TRAILING_STOP_MULTIPLIER)
        }
        
        strategy = VolatilityBreakoutTrailingStrategy(config=strategy_config)
        
        # 3. 리스크 매니저
        risk_manager = RiskManager(
            risk_per_trade=params.get('RISK_PER_TRADE', Config.RISK_PER_TRADE),
            leverage=params.get('LEVERAGE', Config.LEVERAGE)
        )
        
        # 4. 백테스터 실행
        tester = Backtester(df, strategy, risk_manager, symbol=symbol)
        tester.balance = allocated_balance
        
        print(f"   [Process {os.getpid()}] Running {symbol}...")
        report = tester.run()
        
        if report:
            final_bal = tester.balance
            roi = ((final_bal - allocated_balance) / allocated_balance) * 100
            
            return {
                "Symbol": symbol,
                "Init_Bal": allocated_balance,
                "Final_Bal": final_bal,
                "ROI(%)": roi,
                "MDD(%)": report['mdd_pct'],
                "WinRate(%)": report['win_rate_pct'],
                "Trades": report['total_trades'],
                "Liq": report['liquidations'],
                "Safety(%)": report['min_safety_margin']
            }
        else:
            # 거래 없음
            return {
                 "Symbol": symbol,
                 "Init_Bal": allocated_balance,
                 "Final_Bal": allocated_balance,
                 "ROI(%)": 0.0,
                 "MDD(%)": 0.0,
                 "WinRate(%)": 0.0,
                 "Trades": 0,
                 "Liq": 0,
                 "Safety(%)": 0.0
            }
            
    except Exception as e:
        print(f"❌ Error processing {symbol}: {e}")
        import traceback
        traceback.print_exc()
        return { # 에러 발생 시에도 결과 형식 맞춰서 반환 (집계 위함)
            "Symbol": symbol,
            "Init_Bal": allocated_balance,
            "Final_Bal": allocated_balance,
            "ROI(%)": 0.0,
            "MDD(%)": 0.0,
            "WinRate(%)": 0.0,
            "Trades": 0,
            "Liq": 0,
            "Safety(%)": 0.0,
            "Error": str(e)
        }

def run_portfolio_backtest():
    print(f"\n{'='*60}")
    print(f"🚀 AI Crypto Trader - Parallel Portfolio Mode (1h Data)")
    print(f"{'='*60}")

    tasks = []
    
    # 1. 작업 목록 생성
    for symbol, params in Config.PORTFOLIO.items():
        allocation_ratio = params.get('ALLOCATION', 0.0)
        if allocation_ratio <= 0:
            continue
            
        allocated_balance = Config.INITIAL_BALANCE * allocation_ratio
        tasks.append((symbol, params, allocated_balance))

    portfolio_results = []
    total_initial = 0
    total_final = 0

    # 2. 병렬 실행 (CPU 코어 수 활용)
    # max_workers=None이면 os.cpu_count() 만큼 생성됨
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(run_single_backtest, tasks))

    # 3. 결과 집계
    for res in results:
        if res: # 결과가 None이 아닌 경우만 처리 (실패한 프로세스는 None 반환할 수 있음)
            portfolio_results.append(res)
            total_initial += res['Init_Bal']
            total_final += res['Final_Bal']

    # 4. 최종 리포트
    print(f"\n\n{'='*80}")
    print(f"🏆 PORTFOLIO FINAL REPORT")
    print(f"{'='*80}")
    
    if portfolio_results:
        df_res = pd.DataFrame(portfolio_results)
        cols = ["Symbol", "Init_Bal", "Final_Bal", "ROI(%)", "MDD(%)", "WinRate(%)", "Trades", "Liq", "Safety(%)"]
        # Error 컬럼이 있을 경우에만 출력에 포함
        if "Error" in df_res.columns:
            cols.append("Error")
        print(df_res[cols].to_string(index=False, float_format="%.2f"))
        
        print("-" * 80)
        
        total_profit = total_final - total_initial
        total_roi = (total_profit / total_initial) * 100 if total_initial > 0 else 0
        
        print(f"TOTAL INITIAL: ${total_initial:,.2f}")
        print(f"TOTAL FINAL:   ${total_final:,.2f}")
        print(f"NET PROFIT:    ${total_profit:,.2f}")
        print(f"TOTAL ROI:     {total_roi:.2f}%")
    else:
        print("No simulation results available.")
    
    print(f"{'='*80}")

if __name__ == "__main__":
    # Windows/macOS의 'spawn' 방식 프로세스 생성을 위한 안전 가드
    try:
        run_portfolio_backtest()
    except KeyboardInterrupt:
        print("\n🛑 Aborted by user.")
    except Exception as e:
        print(f"\n❌ Critical Error: {e}")
        import traceback
        traceback.print_exc()
