import pandas as pd
import time
import os
import sys
import itertools 
import numpy as np
from threading import Lock # [필수] 멀티스레드 파일 충돌 방지

# 프로젝트 루트 경로를 sys.path에 추가
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from config.settings import Config
from src.data_loader import DataLoader
from src.backtester import Backtester
# [수정] 트레일링 스탑 전략 사용
from src.strategies.volatility_breakout_trailing import VolatilityBreakoutTrailingStrategy
from src.risk_manager import RiskManager

# --- 전역 변수: CSV 쓰기 잠금 장치 ---
# 여러 스레드가 동시에 파일에 쓰려고 할 때 줄을 세우는 역할
csv_write_lock = Lock()

# --- 파라미터 조합 정의 ---

# --- [신규] 여기서 테스트하고 싶은 파라미터 조합을 정의합니다 ---
# optimize.py의 search_space와 동일한 개념입니다.
# SEARCH_SPACE = {
#     'test_days': (np.arange(30, 1461, 30)).tolist(),
#     'risk_per_trade': (np.arange(10, 21, 5)/100).tolist(), # 10%~30%
#     'leverage': (np.arange(1, 6, 1)).tolist(),            # 1, 3, 5, 7, 9배
#     'sl_multiplier': [2.0, 3.0, 4.0],
#     'trailing_mult': (np.arange(20, 51, 5)/10).tolist()    # 2.0 ~ 5.0배
# }

SEARCH_SPACE = {
    'test_days': [10],
    'risk_per_trade': [0.02], # 10%~30%
    'leverage': [2],            # 1, 3, 5, 7, 9배
    'sl_multiplier': [2.0, 3.0, 4.0],
    'trailing_mult': [20]    # 2.0 ~ 5.0배
}

def run_experiment(params: dict, description: str):
    """지정된 파라미터로 백테스트를 실행하고 결과를 CSV에 기록합니다."""
    
    # print(f"🚀 Running: {description}") # 너무 시끄러우면 주석 처리

    # 1. Config 값 임시 변경 (이 실행에만 적용)
    original_settings = {
        'TEST_DAYS': Config.TEST_DAYS,
        'RISK_PER_TRADE': Config.RISK_PER_TRADE,
        'LEVERAGE': Config.LEVERAGE,
        'STOP_LOSS_MULTIPLIER': Config.STOP_LOSS_MULTIPLIER,
        'TRAILING_STOP_MULTIPLIER': Config.TRAILING_STOP_MULTIPLIER
    }
    
    Config.TEST_DAYS = params['test_days']
    Config.RISK_PER_TRADE = params['risk_per_trade']
    Config.LEVERAGE = params['leverage']
    Config.STOP_LOSS_MULTIPLIER = params['sl_multiplier']
    Config.TRAILING_STOP_MULTIPLIER = params['trailing_mult']

    # 2. 백테스트 실행
    # 독립성을 위해 매번 객체 생성
    loader = DataLoader()
    df = loader.get_backtest_data(force_update=False)
    
    # 전략 & 리스크 매니저 (Config 값을 읽어감)
    strategy = VolatilityBreakoutTrailingStrategy()
    risk_manager = RiskManager()
    
    # 데이터 충돌 방지를 위해 df.copy() 사용 권장
    tester = Backtester(df.copy(), strategy, risk_manager)
    results_dict = tester.run()

    # 3. 결과 기록 준비
    log_file_path = os.path.join(Config.SAVE_DIR, "experiment_log.csv")

    new_log_entry = {
        "timestamp": pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        "experiment_id": f"exp_{int(time.time())}",
        "description": description,
        "timeframe": Config.TIMEFRAME,
    }
    new_log_entry.update(params)
    
    if results_dict:
        new_log_entry.update(results_dict)
    else:
        new_log_entry.update({"roi_pct": 0, "mdd_pct": 0, "total_trades": 0, "win_rate_pct": 0, "liquidations": 0})

    # 4. CSV 파일 저장 (Thread-Safe)
    # Lock을 걸어서 동시에 여러 스레드가 파일을 건드리지 못하게 함
    with csv_write_lock:
        try:
            # 파일이 없으면 헤더 포함해서 생성
            if not os.path.exists(log_file_path):
                pd.DataFrame([new_log_entry]).to_csv(log_file_path, index=False)
            else:
                # 파일이 있으면 데이터만 추가 (mode='a', header=False)
                pd.DataFrame([new_log_entry]).to_csv(log_file_path, mode='a', header=False, index=False)
        
        except Exception as e:
            print(f"❌ Failed to save experiment log: {e}")

    # Config 원복 (필수)
    for key, value in original_settings.items():
        setattr(Config, key, value)

if __name__ == "__main__":
    from concurrent.futures import ThreadPoolExecutor
    from tqdm import tqdm 

    # 1. 로그 레벨 조정
    original_log_level = Config.LOG_LEVEL
    Config.LOG_LEVEL = "WARNING"
    print(f"Temporarily setting LOG_LEVEL to WARNING.")
    
    # 2. [핵심] 기존 로그 파일 초기화 (덮어쓰기 모드 구현)
    log_file_path = os.path.join(Config.SAVE_DIR, "experiment_log.csv")
    if os.path.exists(log_file_path):
        try:
            os.remove(log_file_path)
            print(f"🗑️  Existing 'experiment_log.csv' deleted. Starting fresh.")
        except Exception as e:
            print(f"⚠️  Could not delete existing log file: {e}")

    # 3. 파라미터 조합 생성
    param_names = list(SEARCH_SPACE.keys())
    param_values = list(SEARCH_SPACE.values())
    all_combinations = list(itertools.product(*param_values))
    total_combinations = len(all_combinations)
    
    print(f"\n{'='*60}")
    print(f"🤖 Starting Automated Experiment (Fresh Start)")
    print(f"Total combinations to test: {total_combinations}")
    print(f"{'='*60}\n")

    # 4. 멀티스레딩 실행
    with ThreadPoolExecutor(max_workers=None) as executor:
        futures = []
        for i, combo in enumerate(all_combinations):
            params_to_test = dict(zip(param_names, combo))
            description = f"Combo {i+1}"
            future = executor.submit(run_experiment, params=params_to_test, description=description)
            futures.append(future)
        
        for future in tqdm(futures, total=total_combinations, desc="Processing"):
            try:
                future.result()
            except Exception as e:
                print(f"An error occurred in a thread: {e}")

    # 5. 종료 처리
    Config.LOG_LEVEL = original_log_level
    print(f"\n{'='*60}")
    print(f"🎉 All experiments complete.")
    print(f"Results saved to: {log_file_path}")
    print(f"{'='*60}")