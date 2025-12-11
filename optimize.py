import sys
import os
import itertools
import pandas as pd
import numpy as np  # [추가] NumPy 활용
import multiprocessing
import gc  # [추가] 가비지 컬렉션 제어
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# 프로젝트 모듈 임포트
from config.settings import Config
from src.data_loader import DataLoader
from lib.engine import Engine
from strategies.vbo_strategy import VolatilityBreakoutStrategy

# =========================================================
# 🧪 실험 설정
# =========================================================
TARGET_SYMBOL = "BTC/USDT"
TEST_DAYS_FOR_OPTIMIZATION = 365 

# [메모리 최적화] 전역 변수: DataFrame 대신 'Dictionary of NumPy Arrays'를 저장
# 예: {'open': np.array([...]), 'close': np.array([...]), ...}
_worker_data_dict = None

SEARCH_SPACE = {
    'k_window': [20],
    'sma_period': [20, 30, 50, 60],
    'atr_period': [14],
    'sl_mult': [2.0, 2.5, 3.0, 3.5, 4.0],
    'trailing_mult': [2.0, 3.0, 4.0],
    'risk_per_trade': [0.1, 0.2],
    'leverage': [1.0, 2.0, 3.0]
}

def init_worker(symbol, test_days):
    """
    [Extreme Optimization]
    워커 프로세스 초기화:
    1. 데이터를 로드하여 'Float32'로 변환 (메모리 50% 절감)
    2. Pandas DataFrame을 NumPy Dictionary로 분해 (접근 속도 향상)
    """
    global _worker_data_dict
    
    Config.SYMBOL = symbol
    
    try:
        loader = DataLoader()
        df = loader.get_backtest_data(force_update=False)
        
        if df.empty:
            _worker_data_dict = None
            return

        # 1. 기간 필터링 (tail 사용이 iloc보다 미세하게 빠름)
        rows_needed = test_days * 24
        if len(df) > rows_needed:
            df = df.tail(rows_needed)
            
        # 2. [핵심] Memory Optimization: Float64 -> Float32
        # 금융 데이터에서 float32의 정밀도면 충분합니다. 메모리는 절반으로 줍니다.
        float_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in float_cols:
            if col in df.columns:
                df[col] = df[col].astype('float32')

        # 3. [핵심] Speed Optimization: DataFrame -> Dict of NumPy
        # DataFrame을 통째로 들고 있는 것보다, 컬럼별 NumPy 배열로 쪼개는 것이
        # 멀티프로세싱 메모리 관리와 인덱싱 속도 면에서 유리합니다.
        _worker_data_dict = {
            'index': df.index,  # DatetimeIndex
            'open': df['open'].values,
            'high': df['high'].values,
            'low': df['low'].values,
            'close': df['close'].values,
            'volume': df['volume'].values
        }
        
        # 원본 df는 삭제하여 메모리 해제
        del df
        gc.collect() # 강제 가비지 컬렉션
        
    except Exception as e:
        print(f"⚠️ Worker Initialization Failed: {e}")
        _worker_data_dict = None

def run_single_backtest(params):
    """
    [Extreme Optimized]
    Dictionary of Arrays를 그대로 Engine에 주입 (DataFrame 생성 X)
    """
    global _worker_data_dict
    
    if _worker_data_dict is None:
        return {'error': 'Data not loaded'}

    try:
        # 1. 엔진 구동 (Silent Mode)
        engine = Engine(initial_cash=Config.INITIAL_BALANCE, verbose=False)
        
        # [핵심 변경] DataFrame 재조립 과정 삭제! 
        # DataFeed가 이제 Dict를 바로 받아서 내부 멤버 변수(Arrays)에 꽂아버림.
        # Overhead = 0
        engine.add_data(_worker_data_dict) 
        
        # 2. 전략 설정
        engine.add_strategy(VolatilityBreakoutStrategy, **params)
        
        # 3. 실행
        result = engine.run()
        
        return {
            **params,
            'roi': result['roi_pct'],
            'win_rate': result['win_rate_pct'],
            'trades': result['total_trades'],
            'final_balance': result['final_balance']
        }

    except Exception as e:
        # 에러 디버깅을 위해
        # import traceback
        # traceback.print_exc() 
        return {'error': str(e)}

def main():
    print(f"\n{'='*60}")
    print(f"🧬 Hyperparameter Optimization (Float32 + NumPy)")
    print(f"{'='*60}")

    # 1. 조합 생성
    keys = list(SEARCH_SPACE.keys())
    values = list(SEARCH_SPACE.values())
    combinations = list(itertools.product(*values))
    
    tasks = []
    for combo in combinations:
        param_dict = dict(zip(keys, combo))
        param_dict['symbol'] = TARGET_SYMBOL
        tasks.append(param_dict)
        
    cpu_cores = multiprocessing.cpu_count()
    print(f"👉 Total Combinations: {len(tasks)}")
    print(f"👉 CPU Cores: {cpu_cores}")
    
    results = []
    error_count = 0
    
    # 2. 병렬 처리 실행
    # init_worker를 통해 각 프로세스 메모리에 NumPy 배열을 적재
    with ProcessPoolExecutor(max_workers=cpu_cores, 
                             initializer=init_worker, 
                             initargs=(TARGET_SYMBOL, TEST_DAYS_FOR_OPTIMIZATION)) as executor:
        
        futures = {executor.submit(run_single_backtest, task): task for task in tasks}
        
        for future in tqdm(as_completed(futures), total=len(tasks), desc="Processing"):
            res = future.result()
            if res:
                if 'error' not in res:
                    results.append(res)
                else:
                    error_count += 1
    
    # 3. 결과 처리
    if not results:
        print(f"\n❌ No valid results. Errors: {error_count}")
        return

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(by='roi', ascending=False)
    
    save_path = f"optimization_{TARGET_SYMBOL.replace('/', '_')}.csv"
    df_results.to_csv(save_path, index=False)
    
    print(f"\n✅ Optimization Complete!")
    print(f"💾 Saved: {save_path}")
    print(f"\n🏆 Top 5 Parameters:")
    print(df_results.head(5).to_string(index=False))

    df_filtered = df_results[df_results['trades'] >= 30]
    if not df_filtered.empty:
        print(f"\n✨ Recommended (Trades >= 30):")
        print(df_filtered.head(3).to_string(index=False))

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()