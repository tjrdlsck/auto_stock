import sys
import os
import json

# 프로젝트 루트 경로 추가 (모듈 임포트를 위해)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import shutil
import pandas as pd
import numpy as np
import joblib
from hmmlearn.hmm import GaussianHMM
from config import CONFIG, DATA_DIR, MODELS_DIR

# [NEW] 중앙 집중식 피처 엔지니어링 함수 임포트
from alpha_layer.features import apply_features

class Brain:
    def __init__(self):
        self.n_components = 3  # 3개 국면 (상승, 횡보, 하락)
        # [Phase 10] 최적화: 1000 -> 100회로 단축 (수렴하기에 충분함)
        self.iter = 100       
        # [Phase 10] 최적화: 조기 종료 허용 오차 (변화가 적으면 즉시 중단)
        self.tol = 0.01

    def load_data(self, symbol):
        """CSV 파일 로드 및 전처리 (공통 모듈 사용)"""
        clean_symbol = symbol.replace('/', '')
        file_path = os.path.join(DATA_DIR, f"{clean_symbol}_{CONFIG['TIMEFRAME']}.csv")
        
        if not os.path.exists(file_path):
            print(f"❌ 데이터 파일 없음: {file_path}")
            return None, None

        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        
        # [Modified] 피처 엔지니어링 및 스케일링 통합 수행
        df = apply_features(df)
        
        if df is None or df.empty:
            return None, None

        return df, file_path

    def train_model(self, df, symbol):
        """HMM 모델 학습"""
        # 로그 과다 출력 방지를 위해 print 주석 처리 가능
        # print(f"🧠 [{symbol}] 모델 학습 중... (데이터: {len(df)} rows)")
        
        # Features: Log Returns, Range Volatility, RSI, OBV
        X = df[['Log_Returns_Scaled', 'Range_Vol_Scaled', 'RSI_14_Scaled', 'OBV_Scaled']].values
        
        # Gaussian HMM 설정
        model = GaussianHMM(
            n_components=self.n_components, 
            covariance_type="full", 
            n_iter=self.iter, 
            tol=self.tol,      # [Phase 10] 조기 종료 파라미터 추가
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
        # 나머지 하나
        sideways_idx = [x for x in [0, 1, 2] if x not in [bear_idx, bull_idx]][0]
        
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
                
                # 메타데이터 생성
                save_map = {}
                for reg_id, name in regime_map.items():
                    state_id = int(reg_id)
                    if 'Bull' in name: save_map['bull'] = state_id
                    elif 'Bear' in name: save_map['bear'] = state_id
                    else: save_map['sideways'] = state_id

                # 결과 출력
                print(f"📊 [{symbol} 국면 정의 결과]")
                for idx, name in regime_map.items():
                    avg_ret = stats[idx] * 100
                    print(f"  👉 Regime {idx} ({name}): 평균 수익률 {avg_ret:.4f}%")
                
                # 모델 및 메타데이터 저장
                model_path = os.path.join(MODELS_DIR, f"hmm_{clean_symbol}.pkl")
                meta_path = os.path.join(MODELS_DIR, f"hmm_{clean_symbol}_meta.json")
                
                temp_path = model_path + ".tmp"
                joblib.dump(model, temp_path)
                shutil.move(temp_path, model_path) 
                
                with open(meta_path, 'w', encoding='utf-8') as f:
                    json.dump(save_map, f, indent=4)

                print(f"💾 모델 안전 저장 완료: {model_path}")
                
                # 데이터 업데이트
                df_result.to_csv(file_path)
                print(f"💾 데이터 파일 업데이트 완료.\n")
                
            except Exception as e:
                print(f"❌ 학습 실패 ({symbol}): {e}\n")

if __name__ == "__main__":
    brain = Brain()
    brain.run_training()