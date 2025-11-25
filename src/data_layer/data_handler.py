import sys
import os

# 프로젝트 루트 경로 추가 (모듈 임포트를 위해)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import ccxt
import pandas as pd
import time
from datetime import datetime, timedelta

# [NEW] 중앙 집중식 피처 엔지니어링 함수 임포트
from alpha_layer.features import apply_features
from config import CONFIG, DATA_DIR


class MultiSymbolLoader:
    def __init__(self):
        self.exchange = ccxt.binance({
            'options': {'defaultType': 'future'},
            'enableRateLimit': True,
        })
        self.limit = 1000  # 한 번 요청 시 가져올 캔들 최대 개수

    def fetch_ohlcv(self, symbol, days):
        """단일 심볼에 대한 데이터 수집"""
        # 슬래시(/)가 파일명에 들어가면 안되므로 제거 (BTC/USDT -> BTCUSDT)
        clean_symbol = symbol.replace('/', '')
        file_path = os.path.join(DATA_DIR, f"{clean_symbol}_{CONFIG['TIMEFRAME']}.csv")
        
        print(f"\n📥 [{symbol}] 데이터 수집 시작 ({days}일 치)...")
        
        since = self.exchange.parse8601((datetime.now() - timedelta(days=days)).isoformat())
        all_candles = []
        
        while True:
            try:
                candles = self.exchange.fetch_ohlcv(
                    symbol=symbol, 
                    timeframe=CONFIG['TIMEFRAME'], 
                    since=since, 
                    limit=self.limit
                )
                
                if not candles:
                    break
                
                all_candles.extend(candles)
                last_time = candles[-1][0]
                since = last_time + 1
                
                # 진행 상황 표시
                curr_date = datetime.fromtimestamp(last_time/1000)
                print(f"   Run: {curr_date.strftime('%Y-%m-%d')} 데이터 수신 중...", end='\r')
                
                if last_time >= self.exchange.milliseconds():
                    break
                
                # API 호출 제한 방지
                time.sleep(0.1)
                
            except Exception as e:
                print(f"\n❌ {symbol} 수집 중 에러: {e}")
                break
        
        print(f"\n✅ 수집 완료: 총 {len(all_candles)}개 캔들")
        
        # DataFrame 변환
        df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # 중복 제거
        df = df[~df.index.duplicated(keep='last')]
        
        return df, file_path

    def add_features(self, df):
        """
        [Modified] 중앙 집중식 피처 엔지니어링 적용
        기존의 개별 구현을 제거하고 features.py의 apply_features를 사용합니다.
        """
        return apply_features(df)

    def run_pipeline(self):
        """설정된 모든 심볼에 대해 작업 수행"""
        print("="*50)
        print("🚀 멀티 심볼 데이터 파이프라인 가동")
        print("="*50)

        for symbol in CONFIG['SYMBOLS']:
            # 1. 데이터 수집
            df, file_path = self.fetch_ohlcv(symbol, days=CONFIG['FETCH_DAYS'])
            
            # 2. 피처 엔지니어링
            if not df.empty:
                df = self.add_features(df)
                
                # 3. CSV 저장
                df.to_csv(file_path)
                print(f"💾 저장 완료: {file_path}")
            else:
                print(f"⚠️ {symbol} 데이터가 비어있습니다.")
            
            # 심볼 간 딜레이 (안전장치)
            time.sleep(1)

if __name__ == "__main__":
    loader = MultiSymbolLoader()
    loader.run_pipeline()