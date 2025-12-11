import sys
import os
import itertools
import pandas as pd
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# 프로젝트 모듈 임포트
from config.settings import Config
from src.data_loader import DataLoader
from lib.engine import Engine
from strategies.vbo_strategy import VolatilityBreakoutStrategy

# =========================================================
# 🧪 실험 설정 (Search Space)
# =========================================================
TARGET_SYMBOL = "BTC/USDT"  # 최적화할 종목 선택

TEST_DAYS_FOR_OPTIMIZATION = 365 

# 테스트할 파라미터 그리드 정의
SEARCH_SPACE = {
    'k_window': [20],                   # 노이즈 평균 기간
    'sma_period': [20, 30, 50, 60],     # 추세 판단 이평선
    'atr_period': [14],
    'sl_mult': [2.0, 2.5, 3.0, 3.5, 4.0],  # 손절폭 (ATR 배수)
    'trailing_mult': [2.0, 3.0, 4.0],      # 익절/트레일링폭
    'risk_per_trade': [0.1, 0.2],          # 자금 투입 비중
    'leverage': [1.0, 2.0, 3.0]            # 레버리지
}

def run_single_backtest(params):
    """
    단일 파라미터 조합으로 백테스트 실행 (Worker Process)
    데이터는 DataLoader가 캐싱된 Parquet을 읽으므로 각 프로세스에서 로드해도 빠름.
    """
    try:
        # 1. Config 오버라이드 (현재 프로세스 내에서만 유효)
        # DataLoader가 사용하는 심볼 설정
        Config.SYMBOL = params['symbol'] 
        
        # 2. 데이터 로드
        # (매번 로드하면 I/O 병목이 생길 수 있으나, OS 캐시와 Parquet 덕분에 감당 가능)
        # 더 최적화하려면 부모 프로세스에서 공유 메모리(Shared Memory)를 써야 하지만 복잡도 증가.
        loader = DataLoader()
        df = loader.get_backtest_data(force_update=False)
        
        if df.empty:
            return None

        # 테스트 기간 자르기 (설정된 Test Days)
        target_days = TEST_DAYS_FOR_OPTIMIZATION 
        if len(df) > test_days * 24:
            df = df.iloc[-(test_days * 24):]

        # 3. 엔진 구동 (Silent Mode)
        engine = Engine(initial_cash=Config.INITIAL_BALANCE, verbose=False)
        engine.add_data(df)
        
        # 전략 파라미터 주입
        engine.add_strategy(VolatilityBreakoutStrategy, **params)
        
        # 4. 실행
        result = engine.run()
        
        # 5. 결과 반환 (필요한 정보만)
        output = {
            **params,  # 입력 파라미터 포함
            'roi': result['roi_pct'],
            'win_rate': result['win_rate_pct'],
            'trades': result['total_trades'],
            'final_balance': result['final_balance']
        }
        return output

    except Exception as e:
        # 에러 발생 시 None 반환 혹은 에러 로그
        return {'error': str(e)}

def main():
    print(f"\n{'='*60}")
    print(f"🧬 Hyperparameter Optimization: {TARGET_SYMBOL}")
    print(f"{'='*60}")

    # 1. 조합 생성
    keys = list(SEARCH_SPACE.keys())
    values = list(SEARCH_SPACE.values())
    combinations = list(itertools.product(*values))
    
    # 각 조합에 심볼 정보 추가
    tasks = []
    for combo in combinations:
        param_dict = dict(zip(keys, combo))
        param_dict['symbol'] = TARGET_SYMBOL
        tasks.append(param_dict)
        
    print(f"👉 Total Combinations: {len(tasks)}")
    print(f"👉 CPU Cores: {multiprocessing.cpu_count()}")
    
    # 2. 병렬 처리 실행
    results = []
    
    # max_workers=None -> CPU 코어 수만큼 자동 할당
    with ProcessPoolExecutor() as executor:
        # tqdm으로 진행률 표시
        futures = {executor.submit(run_single_backtest, task): task for task in tasks}
        
        for future in tqdm(as_completed(futures), total=len(tasks), desc="Running Backtests"):
            res = future.result()
            if res and 'error' not in res:
                results.append(res)
    
    # 3. 결과 분석 및 저장
    if not results:
        print("❌ No results found. Check data or errors.")
        return

    df_results = pd.DataFrame(results)
    
    # 정렬: ROI 내림차순
    df_results = df_results.sort_values(by='roi', ascending=False)
    
    # 결과 저장
    save_path = f"optimization_{TARGET_SYMBOL.replace('/', '_')}.csv"
    df_results.to_csv(save_path, index=False)
    
    print(f"\n✅ Optimization Complete!")
    print(f"💾 Results saved to: {save_path}")
    
    # Top 5 출력
    print(f"\n🏆 Top 5 Parameters by ROI:")
    print(df_results.head(5).to_string(index=False))
    
    # 전략적 추천 (거래 횟수가 너무 적은 것 제외)
    # 최소 거래 횟수 필터 (예: 30회 이상)
    df_filtered = df_results[df_results['trades'] >= 30]
    if not df_filtered.empty:
        print(f"\n✨ Recommended (Trades >= 30):")
        print(df_filtered.head(3).to_string(index=False))

if __name__ == "__main__":
    # 윈도우/맥 멀티프로세싱 안전장치
    multiprocessing.freeze_support()
    main()