import sys
import os

# Add the 'src' directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import shutil
import pandas as pd
import numpy as np
import joblib
from hmmlearn.hmm import GaussianHMM
from config import CONFIG, DATA_DIR, MODELS_DIR

class Brain:
    def __init__(self):
        self.n_components = 3  # 3개 국면 (상승, 횡보, 하락)
        self.iter = 1000       # 학습 반복 횟수

    def load_data(self, symbol):
        """CSV 파일 로드 및 전처리"""
        clean_symbol = symbol.replace('/', '')
        file_path = os.path.join(DATA_DIR, f"{clean_symbol}_{CONFIG['TIMEFRAME']}.csv")
        
        if not os.path.exists(file_path):
            print(f"❌ 데이터 파일 없음: {file_path}")
            return None, None

        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        
        # Rolling Scaling (과거 30개 캔들 기준 정규화 -> 미래 참조 방지)
        window = 30
        features = ['Log_Returns', 'Range_Vol', 'RSI']
        
        for col in features:
            rolling_mean = df[col].rolling(window=window).mean()
            rolling_std = df[col].rolling(window=window).std()
            # 0으로 나누기 방지 (+1e-8)
            df[f'{col}_Scaled'] = (df[col] - rolling_mean) / (rolling_std + 1e-8)
        
        df = df.dropna()
        return df, file_path

    def train_model(self, df, symbol):
        """HMM 모델 학습"""
        print(f"🧠 [{symbol}] 모델 학습 중... (데이터: {len(df)} rows)")
        
        X = df[['Log_Returns_Scaled', 'Range_Vol_Scaled', 'RSI_Scaled']].values
        
        # Gaussian HMM 설정
        # covariance_type='full': 각 피처 간의 상관관계까지 학습 (정교함 UP)
        model = GaussianHMM(
            n_components=self.n_components, 
            covariance_type="full", 
            n_iter=self.iter, 
            random_state=42,
            verbose=False
        )
        model.fit(X)
        
        # 현재 국면 예측
        df['Regime'] = model.predict(X)
        
        return model, df

    def identify_regimes(self, df):
        """
        학습된 국면(0,1,2)이 실제 무엇인지(Bull/Bear/Sideways) 정의
        """
        # 국면별 평균 수익률 계산
        stats = df.groupby('Regime')['Log_Returns'].mean()
        
        # 정렬: 수익률이 가장 낮은 것 -> Bear, 가장 높은 것 -> Bull
        sorted_indices = stats.sort_values().index
        
        bear_idx = sorted_indices[0]      # 수익률 꼴찌
        bull_idx = sorted_indices[-1]     # 수익률 1등
        sideways_idx = [x for x in [0, 1, 2] if x not in [bear_idx, bull_idx]][0] # 나머지
        
        regime_map = {
            bull_idx: 'Bull (Long)',
            bear_idx: 'Bear (Short)',
            sideways_idx: 'Sideways (Cash/Grid)'
        }
        
        return regime_map, stats

    def run_training(self):
        print("="*50)
        print("🎓 AI 모델 학습 파이프라인 가동")
        print("="*50)
        
        for symbol in CONFIG['SYMBOLS']:
            clean_symbol = symbol.replace('/', '')
            
            # 1. 데이터 로드
            df, file_path = self.load_data(symbol)
            if df is None: continue
            
            # 2. 학습
            try:
                model, df_result = self.train_model(df, symbol)
                
                # 3. 국면 정의
                regime_map, stats = self.identify_regimes(df_result)
                
                # 결과 출력
                print(f"📊 [{symbol} 국면 정의 결과]")
                for idx, name in regime_map.items():
                    avg_ret = stats[idx] * 100
                    print(f"  👉 Regime {idx} ({name}): 평균 수익률 {avg_ret:.4f}%")
                
                # 4. 모델 저장 (파일명: hmm_BTCUSDT.pkl)
                model_path = os.path.join(MODELS_DIR, f"hmm_{clean_symbol}.pkl")
                temp_path = model_path + ".tmp"
                joblib.dump(model, temp_path)
                # 운영체제 차원에서 파일 이동(rename)은 원자적(Atomic)이라 읽는 도중 깨지지 않음
                shutil.move(temp_path, model_path) 

                print(f"💾 모델 안전 저장 완료: {model_path}")
                
                # 5. 데이터 업데이트 (Regime 컬럼 추가된 CSV 저장)
                df_result.to_csv(file_path)
                print(f"💾 모델 및 데이터 저장 완료.\n")
                
            except Exception as e:
                print(f"❌ 학습 실패 ({symbol}): {e}\n")

if __name__ == "__main__":
    brain = Brain()
    brain.run_training()
