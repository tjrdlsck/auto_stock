import pandas as pd
import numpy as np
from datetime import datetime

class DataFeed:
    """
    Pandas DataFrame을 입력받아 시뮬레이션 엔진에 순차적으로(Bar-by-Bar) 데이터를 공급하는 클래스.
    """
    def __init__(self, df: pd.DataFrame):
        # 데이터프레임 복사 및 필수 컬럼 확인
        self.df = df.copy()
        self._validate_columns()
        
        # 반복(Iteration) 제어 변수
        self.current_idx = -1
        self.total_len = len(self.df)
        
        # 현재 바(Bar)의 데이터 캐싱
        self.date = None
        self.open = 0.0
        self.high = 0.0
        self.low = 0.0
        self.close = 0.0
        self.volume = 0.0
        
        # 전체 데이터 배열 (Numpy로 변환하여 속도 최적화)
        self.dates = self.df.index.to_pydatetime() # DatetimeIndex 가정
        self.opens = self.df['open'].values
        self.highs = self.df['high'].values
        self.lows = self.df['low'].values
        self.closes = self.df['close'].values
        self.volumes = self.df['volume'].values

    def _validate_columns(self):
        required = ['open', 'high', 'low', 'close', 'volume']
        for col in required:
            if col not in self.df.columns:
                raise ValueError(f"DataFrame missing required column: {col}")

    def reset(self):
        """커서를 초기화합니다."""
        self.current_idx = -1
        self.date = None

    def next(self) -> bool:
        """
        다음 바(Bar)로 이동합니다.
        :return: 데이터가 있으면 True, 더 이상 없으면 False
        """
        self.current_idx += 1
        
        if self.current_idx >= self.total_len:
            return False
            
        # 현재 데이터 갱신 (Numpy 배열 접근으로 고속 처리)
        self.date = self.dates[self.current_idx]
        self.open = self.opens[self.current_idx]
        self.high = self.highs[self.current_idx]
        self.low = self.lows[self.current_idx]
        self.close = self.closes[self.current_idx]
        self.volume = self.volumes[self.current_idx]
        
        return True

    def __len__(self):
        return self.total_len

    @property
    def current_bar(self):
        """현재 인덱스의 전체 행 데이터를 Series나 Dict 형태로 반환하고 싶을 때 사용"""
        return {
            'datetime': self.date,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume
        }