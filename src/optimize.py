# --- src/optimize.py ---

import sys
import os
import optuna
import pandas as pd
import warnings
from optuna.samplers import GridSampler
import numpy as np

# 상위 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from config.settings import Config
from src.data_loader import DataLoader
from src.backtester import Backtester
from src.strategies.volatility_breakout_trailing import VolatilityBreakoutTrailingStrategy
from src.risk_manager import RiskManager

# 경고 메시지 무시
warnings.filterwarnings("ignore")

def objective(trial):
    """
    Optuna 목표 함수 (멀티스레드 안전 버전)
    """
    
    # 1. 파라미터 받기
    param_sma_period = trial.suggest_int('sma_period', 10, 200)
    param_k_period = trial.suggest_int('k_period', 10, 100)
    
    param_sl_multiplier = trial.suggest_float('sl_multiplier', 1.5, 4.0, step=0.1)
    
    if Config.USE_TRAILING_STOP:
        param_trailing_mult = trial.suggest_float('trailing_mult', 2.0, 5.0, step=0.5)
    else:
        _ = trial.suggest_float('trailing_mult', 2.0, 5.0) 
        param_trailing_mult = None
        
    param_leverage = trial.suggest_int('leverage', 1, 10)
    param_risk_per_trade = trial.suggest_float('risk_per_trade', 0.01, 0.25, step=0.01)

    # 2. 전략 설정
    strategy_config = {
        'sma_period': param_sma_period,
        'k_period': param_k_period,
        'sl_multiplier': param_sl_multiplier,
        'trailing_mult': param_trailing_mult
    }
    
    try:
        # 3. [중요] Config를 수정하지 않고 객체에 직접 값을 주입합니다.
        strategy = VolatilityBreakoutTrailingStrategy(config=strategy_config)
        
        # RiskManager가 __init__에서 값을 받도록 수정되어 있어야 합니다!
        risk_manager = RiskManager(
            risk_per_trade=param_risk_per_trade,
            leverage=param_leverage
        )
        
        # df.copy()는 필수 (데이터 충돌 방지)
        tester = Backtester(df.copy(), strategy, risk_manager)
        tester.run()
        
        final_balance = tester.balance
        
        if final_balance <= 0:
            return 0.0
            
        return final_balance

    except Exception as e:
        return 0.0

if __name__ == "__main__":
    from optuna.samplers import GridSampler

    Config.LOG_LEVEL = "WARNING"
    
    print(f"🚀 [Grid Search] Starting Parallel Optimization for {Config.SYMBOL}...")
    
    # 데이터 로드
    loader = DataLoader()
    df = loader.get_backtest_data(force_update=False)
    
    if df.empty:
        print("❌ No data found.")
        sys.exit(1)
        
    # 데이터 자르기
    if 'd' in Config.TIMEFRAME:
        needed = Config.TEST_DAYS + 200 
        if len(df) > needed:
            df = df.iloc[-needed:]
            print(f"✂️ Data sliced to last {len(df)} candles for optimization.")

    # -------------------------------------------------------------
    # 검색 공간 정의 (효율적인 범위로 조정됨)
    # -------------------------------------------------------------
    search_space = {
        'sma_period': [50, 100], 
        'k_period': [20],         
        'sl_multiplier': [2.0, 3.0, 4.0],
        'risk_per_trade': (np.arange(5, 21, 5)/100).tolist(), # 10%~30%
        'leverage': (np.arange(1, 20, 1)).tolist(),            # 1, 3, 5, 7, 9배
        'trailing_mult': (np.arange(20, 51, 5)/10).tolist()    # 2.0 ~ 5.0배
    }
    
    total_combinations = 1
    for values in search_space.values():
        total_combinations *= len(values)

    print(f"🧩 Total Combinations: {total_combinations}")

    # 실행
    sampler = GridSampler(search_space)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    
    # [멀티스레딩 활성화] n_jobs=-1
    # 주의: RiskManager 코드가 반드시 수정되어 있어야 합니다.
    print(f"⚡ Running with n_jobs=-1 (Multi-threading)...")
    study.optimize(objective, n_trials=total_combinations, n_jobs=-1, show_progress_bar=True)

    print("\n" + "="*60)
    print("🏆 BEST PARAMETERS FOUND")
    print(f"{'='*60}")
    print(f"Value: ${study.best_value:,.2f}")
    print("Best Params:")
    for key, value in study.best_params.items():
        print(f"  - {key}: {value}")
    print("="*60)