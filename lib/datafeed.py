import pandas as pd
import numpy as np

class DataFeed:
    """
    [Optimized]
    DataFrame 또는 Dictionary(NumPy Arrays)를 입력받아
    시뮬레이션 엔진에 데이터를 공급하는 클래스.
    """
    def __init__(self, data):
        # 입력 데이터가 DataFrame인 경우와 Dictionary인 경우를 분기 처리
        if isinstance(data, pd.DataFrame):
            self._init_from_dataframe(data)
        elif isinstance(data, dict):
            self._init_from_dict(data)
        else:
            raise ValueError("DataFeed requires a DataFrame or a Dictionary of numpy arrays.")

        # 반복 제어 변수
        self.current_idx = -1
        self.total_len = len(self.closes)
        
        # 현재 바 데이터 캐싱 (속도 향상을 위해 직접 멤버 변수로)
        self.date = None
        self.open = 0.0
        self.high = 0.0
        self.low = 0.0
        self.close = 0.0
        self.volume = 0.0

    def _init_from_dataframe(self, df: pd.DataFrame):
        """DataFrame에서 NumPy 배열 추출"""
        # copy()는 메모리를 쓰지만 안전성을 위해 유지 (단일 실행용)
        # float32로 변환하여 메모리 최적화
        self.dates = df.index.to_pydatetime()
        self.opens = df['open'].values.astype('float32')
        self.highs = df['high'].values.astype('float32')
        self.lows = df['low'].values.astype('float32')
        self.closes = df['close'].values.astype('float32')
        self.volumes = df['volume'].values.astype('float32')
        
        # 전략에서 원본 데이터프레임이 필요할 때를 위해 참조 유지 (단일 실행 시 편의성)
        self.df = df 

    def _init_from_dict(self, data_dict: dict):
        """Dictionary에서 NumPy 배열 직접 참조 (Zero-Copy)"""
        self.dates = data_dict['index'] # DatetimeIndex or Array
        self.opens = data_dict['open']
        self.highs = data_dict['high']
        self.lows = data_dict['low']
        self.closes = data_dict['close']
        self.volumes = data_dict['volume']
        
        # 최적화 모드에서는 df 참조를 생성하지 않음 (메모리 절약)
        self.df = None

    def reset(self):
        self.current_idx = -1
        self.date = None

    def next(self) -> bool:
        self.current_idx += 1
        if self.current_idx >= self.total_len:
            return False
            
        # 배열 직접 접근 (메서드 호출 오버헤드 최소화)
        idx = self.current_idx
        self.date = self.dates[idx]
        self.open = self.opens[idx]
        self.high = self.highs[idx]
        self.low = self.lows[idx]
        self.close = self.closes[idx]
        self.volume = self.volumes[idx]
        
        return True
        
    def __len__(self):
        return self.total_len