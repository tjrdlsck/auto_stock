import pandas as pd
import time
import os
import sys
import itertools 
import numpy as np
# [변경] 프로세스 풀 사용 (진짜 병렬 처리)
from concurrent.futures import ProcessPoolExecutor, as_completed

# 프로젝트 루트 경로를 sys.path에 추가
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from config.settings import Config
from src.data_loader import DataLoader
from src.backtester import Backtester
from src.strategies.volatility_breakout_trailing import VolatilityBreakoutTrailingStrategy
from src.risk_manager import RiskManager

# --- [신규] 여기서 테스트하고 싶은 파라미터 조합을 정의합니다 ---
# optimize.py의 search_space와 동일한 개념입니다.
SEARCH_SPACE = {
    'test_days': (np.arange(30, 1461, 180)).tolist(),
    # 2. 로직 (새로 추가됨!): 종목별 맞춤 옷을 찾기 위함
    'sma_period': [30, 50, 70, 100],
    'k_window': [20], # K는 20으로 고정해도 무방 (너무 많으면 계산 오래 걸림)
    'risk_per_trade': (np.arange(10, 31, 5)/100).tolist(), # 10%~30%
    'leverage': (np.arange(1, 11, 1)).tolist(),            # 1, 3, 5, 7, 9배
    'sl_multiplier': [2.0, 3.0, 4.0],
    'trailing_mult': (np.arange(20, 51, 5)/10).tolist()    # 2.0 ~ 5.0배
}

# SEARCH_SPACE = {
#     'test_days': [10],
#     'risk_per_trade': [0.02], # 10%~30%
#     'leverage': [2],            # 1, 3, 5, 7, 9배
#     'sl_multiplier': [2.0, 3.0, 4.0],
#     'trailing_mult': [20]    # 2.0 ~ 5.0배
# }

def run_experiment(params: dict, description: str):
    """
    [Worker Process]
    지정된 파라미터로 백테스트를 실행하고 '결과 딕셔너리'를 반환합니다.
    (파일 저장은 여기서 하지 않습니다)
    """
    
    # 1. 프로세스별 독립적인 Config 설정
    # (멀티프로세싱은 메모리가 복사되므로 여기서 바꿔도 다른 프로세스에 영향 없음)
    Config.TEST_DAYS = params['test_days']
    Config.RISK_PER_TRADE = params['risk_per_trade']
    Config.LEVERAGE = params['leverage']
    Config.STOP_LOSS_MULTIPLIER = params['sl_multiplier']
    Config.TRAILING_STOP_MULTIPLIER = params['trailing_mult']
    
    # [NEW] 로직 파라미터도 Config에 반영해야 전략이 바뀝니다!
    Config.SMA_PERIOD = params['sma_period']
    Config.VBO_K_WINDOW = params['k_window']

    # 로깅 끄기 (프로세스마다 출력하면 콘솔 꼬임)
    Config.LOG_LEVEL = "WARNING"

    try:
        # 2. 백테스트 실행
        # 데이터 로더는 매번 로드하면 느리므로, 가능하다면 main에서 넘겨주는 게 좋지만
        # 코드 단순화를 위해 여기서 로드 (OS 캐시 덕분에 두 번째부턴 빠름)
        loader = DataLoader()
        df = loader.get_backtest_data(force_update=False)
        
        # 데이터 자르기 (optimize.py와 동일 로직)
        if 'd' in Config.TIMEFRAME:
            needed = Config.TEST_DAYS + 200
            if len(df) > needed:
                df = df.iloc[-needed:]

        strategy = VolatilityBreakoutTrailingStrategy()
        risk_manager = RiskManager()
        
        tester = Backtester(df.copy(), strategy, risk_manager)
        results_dict = tester.run()

        # 3. 결과 데이터 생성
        result_entry = {
            "timestamp": pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            "description": description,
            "timeframe": Config.TIMEFRAME,
        }
        result_entry.update(params)
        
        if results_dict:
            result_entry.update(results_dict)
        else:
            result_entry.update({"roi_pct": 0, "mdd_pct": 0, "total_trades": 0, "win_rate_pct": 0, "liquidations": 0})
            
        return result_entry

    except Exception as e:
        # 에러 발생 시 에러 메시지 반환
        return {"error": str(e), "description": description}

if __name__ == "__main__":
    from tqdm import tqdm 

    # 1. 기존 로그 파일 초기화
    log_file_path = os.path.join(Config.SAVE_DIR, "experiment_log.csv")
    if os.path.exists(log_file_path):
        try:
            os.remove(log_file_path)
            print(f"🗑️  Existing 'experiment_log.csv' deleted. Starting fresh.")
        except Exception as e:
            print(f"⚠️  Could not delete existing log file: {e}")

    # 2. 파라미터 조합 생성
    param_names = list(SEARCH_SPACE.keys())
    param_values = list(SEARCH_SPACE.values())
    all_combinations = list(itertools.product(*param_values))
    total_combinations = len(all_combinations)
    
    print(f"\n{'='*60}")
    print(f"🚀 Starting Multi-Core Experiment (True Parallelism)")
    print(f"Total combinations: {total_combinations}")
    print(f"Using CPU Cores: {os.cpu_count()}") # 사용 가능한 코어 수 출력
    print(f"{'='*60}\n")

    # 3. 멀티 프로세싱 실행
    results_to_save = []
    
    # max_workers=None -> CPU 코어 수만큼 프로세스 생성
    with ProcessPoolExecutor(max_workers=None) as executor:
        futures = []
        for i, combo in enumerate(all_combinations):
            params_to_test = dict(zip(param_names, combo))
            description = f"Combo {i+1}"
            
            # 작업을 프로세스 풀에 던짐
            future = executor.submit(run_experiment, params_to_test, description)
            futures.append(future)
        
        # 4. 결과 수집 및 저장 (메인 프로세스가 담당)
        # as_completed: 끝나는 순서대로 바로바로 처리
        for future in tqdm(as_completed(futures), total=total_combinations, desc="Processing"):
            try:
                data = future.result()
                
                if "error" in data:
                    print(f"❌ Error in {data['description']}: {data['error']}")
                    continue
                
                # 파일에 즉시 기록 (데이터 유실 방지)
                df_row = pd.DataFrame([data])
                if not os.path.exists(log_file_path):
                    df_row.to_csv(log_file_path, index=False)
                else:
                    df_row.to_csv(log_file_path, mode='a', header=False, index=False)
                    
            except Exception as e:
                print(f"CRITICAL ERROR: {e}")

    print(f"\n{'='*60}")
    print(f"🎉 All experiments complete.")
    print(f"Results saved to: {log_file_path}")
    print(f"{'='*60}")