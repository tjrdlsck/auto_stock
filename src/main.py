import sys
import os
# 1. 현재 파일(main.py)의 위치를 기준으로 상위 폴더(src)의 상위 폴더(deep_stock)를 찾음
current_dir = os.path.dirname(os.path.abspath(__file__)) # src 폴더
root_dir = os.path.dirname(current_dir)                  # deep_stock 폴더

# 2. 프로젝트 루트 경로를 sys.path에 추가
sys.path.append(root_dir)

from config.settings import Config
from src.data_loader import DataLoader
from src.backtester import Backtester
# 전략 import 경로 주의
from src.strategies.volatility_breakout_trailing import VolatilityBreakoutTrailingStrategy
from src.risk_manager import RiskManager

def run_backtest():
    """
    [Mode 1] 백테스팅 모드 실행
    과거 데이터를 불러와 시뮬레이션을 수행합니다.
    """
    print(f"\n{'='*60}")
    print(f"🕰️  AI Crypto Trader - Backtest Mode ({Config.TIMEFRAME})")
    print(f"{'='*60}")
    
    # 1. 데이터 로더 초기화
    loader = DataLoader()
    
    # 2. 백테스팅 데이터 준비 (6개월치 수집 -> CSV 저장/로드)
    # force_update=False: 이미 받아둔 CSV가 있으면 그것을 사용 (속도 향상)
    # force_update=True: 강제로 거래소에서 새로 받아옴 (최신 데이터 필요 시)
    df = loader.get_backtest_data(force_update=False)
    
    if df.empty:
        print("❌ [Error] No data available for backtesting.")
        return

    # 3. 전략 및 리스크 매니저 초기화
    strategy = VolatilityBreakoutTrailingStrategy()  # Instantiate the concrete strategy
    risk_manager = RiskManager() # Instantiate the risk manager

    # 4. 백테스터 엔진 초기화 및 실행 (전략과 리스크 매니저 주입)
    tester = Backtester(df=df, strategy=strategy, risk_manager=risk_manager) # Modified
    report = tester.run()

# Removed run_live_trader function as per user request

if __name__ == "__main__":
    # 설정 검증
    try:
        Config.validate()
        run_backtest()
    except Exception as e:
        print(f"❌ Configuration Error: {e}")
        sys.exit(1)
