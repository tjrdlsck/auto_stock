import ccxt
import pandas as pd
import time
import sys
import os
from datetime import datetime, timedelta

# 상위 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from config.settings import Config

class DataLoader:
    """
    거래소 API를 통해 1시간봉(1h) 데이터를 수집하여 저장하는 클래스
    """
    def __init__(self):
        self.exchange_id = Config.EXCHANGE_NAME
        self.symbol = Config.SYMBOL
        self.timeframe = Config.TIMEFRAME # '1h'
        self.data_dir = Config.SAVE_DIR
        
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            
        # 파일명 (기본은 Parquet)
        clean_symbol = self.symbol.replace("/", "_")
        self.parquet_path = os.path.join(self.data_dir, f"{clean_symbol}_{self.timeframe}.parquet")
        self.csv_path = os.path.join(self.data_dir, f"{clean_symbol}_{self.timeframe}.csv")

        try:
            exchange_class = getattr(ccxt, self.exchange_id)
            self.exchange = exchange_class({
                'apiKey': Config.BINANCE_API_KEY,
                'secret': Config.BINANCE_SECRET_KEY,
                'enableRateLimit': True,
                'options': {'defaultType': 'future'}
            })
        except Exception as e:
            print(f"⚠️ Error initializing exchange: {e}")
            sys.exit(1)

    def get_backtest_data(self, force_update=False):
        """
        데이터 로드 또는 갱신 (Parquet 우선)
        """
        # 1. Parquet 확인
        if os.path.exists(self.parquet_path) and not force_update:
            print(f"[DataLoader] Found local Parquet cache: {self.parquet_path}")
            return pd.read_parquet(self.parquet_path)
            
        # 2. CSV 확인 (Migration)
        elif os.path.exists(self.csv_path) and not force_update:
            print(f"[DataLoader] Found local CSV cache, migrating to Parquet...")
            dtypes = {
                'open': 'float32', 'high': 'float32', 'low': 'float32', 
                'close': 'float32', 'volume': 'float32'
            }
            df = pd.read_csv(self.csv_path, index_col='datetime', parse_dates=True, dtype=dtypes)
            
            # Parquet으로 저장 후 리턴
            df.to_parquet(self.parquet_path, engine='pyarrow', compression='snappy')
            print(f"[DataLoader] Migrated to {self.parquet_path}")
            return df

        # 3. 없으면 다운로드
        else:
            print(f"[DataLoader] Fetching new data ({self.timeframe})... This might take a while.")
            return self.fetch_history()

    def fetch_history(self):
        """
        대량의 1시간봉 데이터 수집
        """
        days = Config.TOTAL_DAYS_TO_FETCH
        print(f"[DataLoader] Fetching {days} days of {self.timeframe} data...")
        
        start_date = datetime.now() - timedelta(days=days)
        since = int(start_date.timestamp() * 1000)
        
        all_candles = []
        retry_count = 0
        
        while True:
            try:
                candles = self.exchange.fetch_ohlcv(
                    symbol=self.symbol,
                    timeframe=self.timeframe,
                    since=since,
                    limit=1000 
                )
                
                if not candles:
                    break
                    
                all_candles.extend(candles)
                
                last_timestamp = candles[-1][0]
                since = last_timestamp + 1
                
                now_timestamp = int(datetime.now().timestamp() * 1000)
                if last_timestamp >= now_timestamp - (60 * 60 * 1000): 
                    break
                
                # 진행 상황 표시
                fetched_date = datetime.fromtimestamp(last_timestamp/1000).strftime('%Y-%m-%d')
                print(f"    -> Fetched until {fetched_date} ({len(all_candles)} candles)")
                
                time.sleep(self.exchange.rateLimit / 1000)
                
            except Exception as e:
                print(f"⚠️ Fetch Error: {e}")
                retry_count += 1
                if retry_count > 3: break
                time.sleep(2)

        if not all_candles:
            return pd.DataFrame()

        df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('datetime', inplace=True)
        
        cols = ['open', 'high', 'low', 'close', 'volume']
        df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
        # [Memory Optimization] float32로 변환
        df[cols] = df[cols].astype('float32')
        
        # 중복 및 결측 제거
        df = df[~df.index.duplicated(keep='first')]
        df.dropna(inplace=True)
        
        # Parquet으로 저장
        df.to_parquet(self.parquet_path, engine='pyarrow', compression='snappy')
        print(f"[DataLoader] Successfully saved {len(df)} candles to {self.parquet_path}")
        
        return df

if __name__ == "__main__":
    # 테스트 실행
    loader = DataLoader()
    df = loader.get_backtest_data(force_update=True)