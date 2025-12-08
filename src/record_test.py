import pandas as pd
import time
import os
import sys
import itertools 
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm 

# 프로젝트 루트 경로를 sys.path에 추가
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from config.settings import Config
from src.data_loader import DataLoader
from src.backtester import Backtester
# [중요] 우리가 수정한 1h -> 1d 변환 전략 클래스 import
from src.strategies.volatility_breakout_trailing import VolatilityBreakoutTrailingStrategy
from src.risk_manager import RiskManager

# =========================================================
# 🧪 실험 파라미터 정의 (Search Space)
# Wide SL 전략에 맞춰 범위를 조정했습니다.
# =========================================================
# SEARCH_SPACE = {
#     'test_days': (np.arange(180, 721, 180)).tolist(),
#     # 2. 로직 (새로 추가됨!): 종목별 맞춤 옷을 찾기 위함
#     'sma_period': [30, 50, 70],
#     'k_window': [20], # K는 20으로 고정해도 무방 (너무 많으면 계산 오래 걸림)
#     'risk_per_trade': (np.arange(10, 31, 5)/100).tolist(), # 10%~30%
#     'leverage': (np.arange(1, 10, 2)).tolist(),            # 1, 3, 5, 7, 9배
#     'sl_multiplier': (np.arange(30, 51, 5)/10).tolist(),
#     'trailing_mult': [2.0]     # 2.0 ~ 8.0배
# }

SEARCH_SPACE = {
    # 1. 기간 설정: 최근 1년 ~ 2년 (데이터가 충분해야 함)
    'test_days': [365], 
    
    # 2. 전략 파라미터
    'sma_period': [30, 50, 70],      # 추세 판단 기준
    'k_window': [20],                # Noise Ratio 평균 기간
    
    # 3. 리스크 관리 (Wide SL 핵심)
    'sl_multiplier': [3.0, 4.0, 5.0], # ATR의 3~5배 (널널하게)
    'risk_per_trade': [0.1, 0.2],     # 자산의 10% ~ 20% 투입
    'leverage': [2, 3, 4],            # 레버리지 (저배율 권장)
    
    # 4. 트레일링 스탑
    'trailing_mult': [4.0, 5.0]       # 손절만큼 널널하게 따라감
}

def run_experiment(params: dict, description: str):
    """
    [Worker Process] 개별 백테스트 실행
    """
    try:
        # 1. 프로세스 별 Config 설정 덮어씌우기
        Config.TEST_DAYS = params['test_days']
        Config.SMA_PERIOD = params['sma_period']
        Config.VBO_K_WINDOW = params['k_window']
        
        Config.STOP_LOSS_MULTIPLIER = params['sl_multiplier']
        Config.RISK_PER_TRADE = params['risk_per_trade']
        Config.LEVERAGE = params['leverage']
        Config.TRAILING_STOP_MULTIPLIER = params['trailing_mult']
        
        # [중요] 타임프레임 강제 고정
        Config.TIMEFRAME = "1h"
        Config.LOG_LEVEL = "WARNING" 

        # 2. 데이터 로드 (OS 캐싱 활용)
        loader = DataLoader()
        df = loader.get_backtest_data(force_update=False)
        
        if df.empty:
            return {"error": "No Data", "description": description}

        # 3. 객체 생성
        strategy_config = {
            'sma_period': params['sma_period'],
            'k_window': params['k_window'],
            'atr_period': Config.ATR_PERIOD,
            'sl_multiplier': params['sl_multiplier'],
            'trailing_mult': params['trailing_mult']
        }
        strategy = VolatilityBreakoutTrailingStrategy(config=strategy_config)
        
        risk_manager = RiskManager(
            risk_per_trade=params['risk_per_trade'],
            leverage=params['leverage']
        )
        
        # [수정된 부분] symbol 인자를 꼭 넘겨줘야 함!
        # Config.SYMBOL은 settings.py에서 설정한 값 (예: SOL/USDT)
        tester = Backtester(df.copy(), strategy, risk_manager, symbol=Config.SYMBOL)
        
        results = tester.run()

        if results is None:
            return {"error": "Backtest Failed (Not enough data?)", "description": description}

        # 4. 결과 정리
        result_entry = {
            "timestamp": pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            "description": description,
        }
        result_entry.update(params)      
        result_entry.update(results)     
        
        return result_entry

    except Exception as e:
        # 에러 발생 시 내용을 명확히 반환
        return {"error": str(e), "description": description}

if __name__ == "__main__":
    # 1. 기존 로그 파일 관리
    log_file_path = os.path.join(Config.SAVE_DIR, "experiment_log.csv")
    
    # [Tip] SOL 실험 시작 전 기존 로그 지우기 (선택 사항)
    # 섞이는 게 싫다면 아래 주석을 풀고 사용하세요.
    # if os.path.exists(log_file_path):
    #     try:
    #         os.remove(log_file_path)
    #         print("🗑️ Deleted old experiment log.")
    #     except: pass

    if os.path.exists(log_file_path):
        print(f"♻️  Appending to: {log_file_path}")
    else:
        print(f"🆕 Creating new: {log_file_path}")

    # 2. 조합 생성
    param_names = list(SEARCH_SPACE.keys())
    param_values = list(SEARCH_SPACE.values())
    all_combinations = list(itertools.product(*param_values))
    total_combinations = len(all_combinations)
    
    print(f"\n{'='*60}")
    print(f"🚀 Experiment Target: {Config.SYMBOL}")
    print(f"🎯 Total Combinations: {total_combinations}")
    print(f"💻 CPU Cores: {os.cpu_count()}")
    print(f"{'='*60}\n")

    # 3. 병렬 처리 실행
    with ProcessPoolExecutor(max_workers=None) as executor:
        futures = []
        for i, combo in enumerate(all_combinations):
            params_to_test = dict(zip(param_names, combo))
            description = f"Exp_{i+1}"
            future = executor.submit(run_experiment, params_to_test, description)
            futures.append(future)
        
        # 4. 결과 수집
        write_header = not os.path.exists(log_file_path)
        success_count = 0
        
        for future in tqdm(as_completed(futures), total=total_combinations, desc="Simulating"):
            try:
                data = future.result()
                
                # [수정] 에러가 있으면 화면에 출력해서 원인을 파악하게 함
                if "error" in data:
                    # 너무 많은 에러가 뜨면 콘솔이 지저분해지므로, 첫 번째 에러만 보고 싶다면 주석 처리
                    # 하지만 지금은 원인 파악이 중요하므로 출력 권장
                    # print(f"❌ {data['description']}: {data['error']}")
                    continue
                
                df_row = pd.DataFrame([data])
                df_row.to_csv(log_file_path, mode='a', header=write_header, index=False)
                write_header = False 
                success_count += 1
                    
            except Exception as e:
                print(f"CRITICAL ERROR: {e}")

    print(f"\n{'='*60}")
    print(f"🎉 Experiments Completed: {success_count}/{total_combinations}")
    
    if success_count == 0:
        print("⚠️ Warning: 0 successes. Check if Config.SYMBOL is correct or data exists.")
    else:
        print(f"📄 Results saved to: {log_file_path}")
    print(f"{'='*60}")