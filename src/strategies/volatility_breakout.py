# --- src/strategies/volatility_breakout.py ---

import pandas as pd
import ta
import numpy as np
from typing import Dict, Any

from src.strategies.base import BaseStrategy, TradeSignal
from config.settings import Config

class VolatilityBreakoutStrategy(BaseStrategy):
    """
    변동성 돌파 전략 (Advanced)
    - 특징: Dynamic K, 추세 필터, ATR 손절
    - 개선: 종가 진입이 아닌 장중 돌파(Touch) 시뮬레이션 적용
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        # Config 클래스에서 값 로드
        self.sma_period = Config.SMA_PERIOD
        self.atr_period = Config.ATR_PERIOD
        self.k_window = Config.VBO_K_WINDOW
        self.sl_multiplier = Config.STOP_LOSS_MULTIPLIER
        
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # 1. 전일 데이터 생성 (Look-ahead Bias 방지)
        # 오늘의 목표가는 '어제' 데이터로 계산해야 함
        df['prev_high'] = df['high'].shift(1)
        df['prev_low'] = df['low'].shift(1)
        df['prev_close'] = df['close'].shift(1)
        df['prev_open'] = df['open'].shift(1)
        
        # 2. Range 계산
        df['range'] = df['prev_high'] - df['prev_low']

        # 3. Noise Ratio & Dynamic K
        # Noise = 1 - (실제이동폭 / 전체변동폭)
        noise = 1 - np.abs(df['prev_close'] - df['prev_open']) / (df['range'] + 1e-9)
        # 20일 평균 Noise를 K로 사용
        df['k'] = noise.rolling(window=self.k_window).mean()

        # 4. 목표가(Targets) 미리 계산
        # 오늘 시가 기준 (df['open']은 현재 캔들 시가)
        df['long_target'] = df['open'] + (df['range'] * df['k'])
        df['short_target'] = df['open'] - (df['range'] * df['k'])

        # 5. 보조 지표 (SMA, ATR)
        # 추세 판단은 '어제 종가' 기준이 안전함 (장중 SMA 변동 방지)
        df['sma'] = ta.trend.sma_indicator(df['close'], window=self.sma_period).shift(1)
        df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=self.atr_period).shift(1)

        df.dropna(inplace=True)
        return df

    def generate_signal(self, current_slice: pd.DataFrame) -> TradeSignal:
        if len(current_slice) < 1:
            return TradeSignal(timestamp=None, action="HOLD", entry_price=0, reason="No data")

        # 현재 캔들 (Backtest 시점의 '오늘' 캔들)
        curr = current_slice.iloc[-1]
        
        timestamp = curr.name
        # 현재가(종가), 고가, 저가, 시가
        curr_close = curr['close']
        curr_high = curr['high']
        curr_low = curr['low']
        curr_open = curr['open']

        # 미리 계산된 지표
        long_target = curr['long_target']
        short_target = curr['short_target']
        sma = curr['sma']
        atr = curr['atr']
        
        # --- 논리 시작 ---
        action = "HOLD"
        entry_price = curr_close
        stop_loss = None
        reason = ""

        # 1. Long 진입 판단
        # 조건: 추세 상승(SMA 위) AND 장중 고가가 목표가 터치
        is_bull_trend = curr_close > sma # 혹은 prev_close > sma
        
        if is_bull_trend and (curr_high >= long_target):
            action = "BUY"
            # 진입가는 목표가. 단, 시가가 이미 목표가보다 높으면(Gap 상승) 시가 진입
            entry_price = max(curr_open, long_target)
            stop_loss = entry_price - (atr * self.sl_multiplier)
            reason = f"Long Breakout: Target {long_target:.2f} hit (High {curr_high:.2f})"

        # 2. Short 진입 판단
        # 조건: 추세 하락(SMA 아래) AND 장중 저가가 목표가 터치
        elif (not is_bull_trend) and (curr_low <= short_target):
            action = "SELL"
            # 진입가는 목표가. 단, 시가가 이미 목표가보다 낮으면(Gap 하락) 시가 진입
            entry_price = min(curr_open, short_target)
            stop_loss = entry_price + (atr * self.sl_multiplier)
            reason = f"Short Breakout: Target {short_target:.2f} hit (Low {curr_low:.2f})"
        
        else:
            reason = f"Wait. Price {curr_close:.2f} | Tgt: {long_target:.1f}/{short_target:.1f}"

        return TradeSignal(
            timestamp=timestamp,
            action=action,
            entry_price=entry_price,
            stop_loss=stop_loss,
            reason=reason
        )

# --- Step 2 Debugging / Verification ---
if __name__ == "__main__":
    print("--- Step 2: VolatilityBreakoutStrategy Verification ---")

    # 더미 데이터 생성 (실제 OHLCV 데이터와 유사하게)
    data = {
        'timestamp': pd.to_datetime(pd.date_range(start='2023-01-01', periods=100, freq='D')),
        'open': np.random.uniform(9000, 11000, 100),
        'high': np.random.uniform(11000, 12000, 100),
        'low': np.random.uniform(8000, 9000, 100),
        'close': np.random.uniform(9500, 11500, 100),
        'volume': np.random.uniform(1000, 5000, 100)
    }
    dummy_df = pd.DataFrame(data).set_index('timestamp')
    dummy_df = dummy_df.sort_index()

    # 전략 인스턴스 생성
    strategy = VolatilityBreakoutStrategy()

    # 지표 계산 (전체 데이터에 대해)
    print("\nCalculating indicators...")
    df_with_indicators = strategy.calculate_indicators(dummy_df)
    print(df_with_indicators.head())
    print(df_with_indicators.tail())

    # 신호 생성 테스트 (마지막 몇 개의 데이터 슬라이스에 대해)
    print("\nGenerating signals for last few data points...")
    test_signals = []
    # 충분한 데이터가 있어야 지표가 계산되므로 최소 window 크기 이상 슬라이싱
    min_data_for_signal = max(strategy.sma_period, strategy.atr_period, strategy.k_period) + 2
    
    if len(df_with_indicators) > min_data_for_signal:
        for i in range(min_data_for_signal, len(df_with_indicators)):
            current_slice = df_with_indicators.iloc[:i+1]
            signal = strategy.generate_signal(current_slice)
            if signal.action != "HOLD": # HOLD가 아닌 신호만 출력
                test_signals.append(signal)
                print(f"Signal: {signal}")
    else:
        print("Not enough data to generate signals even after indicator calculation.")

    if not test_signals:
        print("\nNo BUY/SELL signals generated in this dummy run. This is normal for random data.")
    
    print("\n--- Verification Complete ---")