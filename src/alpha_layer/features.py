import pandas as pd
import pandas_ta as ta
import numpy as np

def apply_features(df):
    """
    [중앙 집중식 피처 엔지니어링 함수]
    
    DataHandler(수집), Brain(학습), Trader(실매매)에서 
    동일하게 사용하는 전처리 로직입니다.
    
    Args:
        df (pd.DataFrame): OHLCV 데이터프레임
        
    Returns:
        pd.DataFrame: 보조지표 및 스케일링된 피처가 추가된 데이터프레임
    """
    if df is None or df.empty:
        return df

    # 복사본 생성 (원본 오염 방지)
    df = df.copy()

    # ---------------------------------------
    # 1. 기본 보조지표 계산 (pandas_ta 사용)
    # ---------------------------------------
    
    # RSI (14) -> 컬럼명: RSI_14
    df.ta.rsi(length=14, append=True)
    
    # EMA (20) -> 컬럼명: EMA_20
    df.ta.ema(length=20, append=True)
    
    # ATR (14) -> 컬럼명: ATRr_14 (Running ATR)
    df.ta.atr(length=14, append=True)
    
    # OBV -> 컬럼명: OBV
    df.ta.obv(append=True)

    # ---------------------------------------
    # 2. 커스텀 지표 계산
    # ---------------------------------------
    
    # 로그 수익률 (Log Returns)
    df['Log_Returns'] = np.log(df['close'] / df['close'].shift(1))
    
    # 범위 변동성 (Range Volatility)
    df['Range_Vol'] = (df['high'] - df['low']) / df['close']

    # ---------------------------------------
    # 3. 롤링 스케일링 (Rolling Z-Score)
    # AI 모델 입력용 (Lookahead Bias 방지)
    # ---------------------------------------
    window = 30
    features_to_scale = ['Log_Returns', 'Range_Vol', 'RSI_14', 'OBV']
    
    for col in features_to_scale:
        # 해당 컬럼이 존재하는지 확인
        if col in df.columns:
            rolling_mean = df[col].rolling(window=window).mean()
            rolling_std = df[col].rolling(window=window).std()
            
            # 0으로 나누기 방지 (+ 1e-8)
            df[f'{col}_Scaled'] = (df[col] - rolling_mean) / (rolling_std + 1e-8)

    # ---------------------------------------
    # 4. 데이터 정제
    # ---------------------------------------
    # NaN 값 제거 (초기 윈도우 구간)
    df.dropna(inplace=True)
    
    # 무한대 값 제거 (Zero Division 등)
    df.replace([np.inf, -np.inf], 0, inplace=True)

    return df