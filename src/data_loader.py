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
    거래소 API를 통해 대량의 과거 데이터를 수집하고 관리하는 클래스
    """
    def __init__(self):
        self.exchange_id = Config.EXCHANGE_NAME
        self.symbol = Config.SYMBOL
        self.timeframe = Config.TIMEFRAME
        self.data_dir = Config.SAVE_DIR
        
        # 저장 경로 생성 (data 폴더)
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            
        # 파일명 생성 (예: data/BTC_USDT_1h.csv)
        clean_symbol = self.symbol.replace("/", "_")
        self.file_path = os.path.join(self.data_dir, f"{clean_symbol}_{self.timeframe}.csv")

        # CCXT 거래소 객체 초기화
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
        백테스팅용 데이터를 준비합니다.
        로컬 파일이 있으면 로드하고, 없거나 force_update=True면 새로 수집합니다.
        """
        if os.path.exists(self.file_path) and not force_update:
            print(f"[DataLoader] Found local cache: {self.file_path}")
            df = pd.read_csv(self.file_path, index_col='datetime', parse_dates=True)
            
            # 데이터가 충분한지 확인 (설정된 기간보다 짧으면 새로 받기 권장)
            # 여기서는 단순 로드
            print(f"[DataLoader] Loaded {len(df)} candles from cache.")
            return df
        else:
            print(f"[DataLoader] No local cache found (or forced update). Fetching from exchange...")
            return self.fetch_history()

    def fetch_history(self):
        """
        설정된 기간(TOTAL_DAYS_TO_FETCH)만큼의 데이터를 반복 수집합니다.
        """
        days = Config.TOTAL_DAYS_TO_FETCH
        print(f"[DataLoader] Fetching historical data for the last {days} days...")
        
        # 수집 시작 시간 계산 (밀리초 단위)
        start_date = datetime.now() - timedelta(days=days)
        since = int(start_date.timestamp() * 1000)
        
        all_candles = []
        retry_count = 0
        
        while True:
            try:
                # 데이터 가져오기 (최대 1000개)
                candles = self.exchange.fetch_ohlcv(
                    symbol=self.symbol,
                    timeframe=self.timeframe,
                    since=since,
                    limit=1000 
                )
                
                if not candles:
                    break
                    
                all_candles.extend(candles)
                
                # 다음 요청을 위해 시간 업데이트 (마지막 캔들 시간 + 1ms)
                last_timestamp = candles[-1][0]
                since = last_timestamp + 1
                
                # 현재 시간까지 다 가져왔으면 종료
                now_timestamp = int(datetime.now().timestamp() * 1000)
                if last_timestamp >= now_timestamp - (60 * 60 * 1000): # 1시간 이내면 중단
                    break
                
                print(f"    Fetched {len(candles)} candles... (Last: {datetime.fromtimestamp(last_timestamp/1000)})")
                time.sleep(self.exchange.rateLimit / 1000) # Rate Limit 준수
                
            except Exception as e:
                print(f"⚠️ Fetch Error: {e}")
                retry_count += 1
                if retry_count > 3:
                    break
                time.sleep(2)

        # DataFrame 변환
        if not all_candles:
            print("❌ Failed to fetch data.")
            return pd.DataFrame()

        df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('datetime', inplace=True)
        
        # 숫자형 변환
        cols = ['open', 'high', 'low', 'close', 'volume']
        df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
        
        # 중복 제거 (API 호출 경계에서 중복 발생 가능성)
        df = df[~df.index.duplicated(keep='first')]
        
        # CSV 저장
        df.to_csv(self.file_path)
        print(f"[DataLoader] Saved {len(df)} candles to {self.file_path}")
        
        return df

if __name__ == "__main__":
    loader = DataLoader()
    # 테스트: 데이터 수집 실행
    df = loader.get_backtest_data(force_update=True)
    print(df.head())
    print(df.tail())