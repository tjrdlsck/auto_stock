import pandas as pd
import ta
import numpy as np
from typing import Dict, Any

from src.strategies.base import BaseStrategy, TradeSignal
from config.settings import Config

class VolatilityBreakoutTrailingStrategy(BaseStrategy):
    """
    변동성 돌파 + 트레일링 스탑 (ATR 기반)
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        
        # 1. 기존 파라미터 로드
        self.sma_period = self.config.get('sma_period', Config.SMA_PERIOD)
        self.k_window = self.config.get('k_window', Config.VBO_K_WINDOW)
        self.atr_period = self.config.get('atr_period', Config.ATR_PERIOD)
        self.sl_multiplier = self.config.get('sl_multiplier', Config.STOP_LOSS_MULTIPLIER)
        
        # 2. [핵심] 트레일링 스탑 설정 로드
        # 우선순위: (1) 최적화 파라미터 -> (2) Settings.py 설정
        
        # 만약 Settings에서 사용 안 함으로 되어 있으면 None 처리
        if not Config.USE_TRAILING_STOP:
             self.trailing_mult = None
        else:
            # Config에서 기본값 가져오기
            default_trailing = Config.TRAILING_STOP_MULTIPLIER
            
            # 최적화 중이라면 config 딕셔너리에 있는 값을 우선 사용 (없으면 default 사용)
            self.trailing_mult = self.config.get('trailing_mult', default_trailing)

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        # 기존과 100% 동일
        df = df.copy()
        df['prev_high'] = df['high'].shift(1)
        df['prev_low'] = df['low'].shift(1)
        df['prev_close'] = df['close'].shift(1)
        df['prev_open'] = df['open'].shift(1)
        
        df['range'] = df['prev_high'] - df['prev_low']
        noise = 1 - np.abs(df['prev_close'] - df['prev_open']) / (df['range'] + 1e-9)
        df['k'] = noise.rolling(window=self.k_window).mean()

        df['long_target'] = df['open'] + (df['range'] * df['k'])
        df['short_target'] = df['open'] - (df['range'] * df['k'])

        df['sma'] = ta.trend.sma_indicator(df['close'], window=self.sma_period).shift(1)
        df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=self.atr_period).shift(1)

        df.dropna(inplace=True)
        return df

    def generate_signal(self, current_slice: pd.DataFrame) -> TradeSignal:
        if len(current_slice) < 1:
            return TradeSignal(timestamp=None, action="HOLD", entry_price=0)

        curr = current_slice.iloc[-1]
        timestamp = curr.name
        curr_close = curr['close']
        curr_high = curr['high']
        curr_low = curr['low']
        curr_open = curr['open']

        long_target = curr['long_target']
        short_target = curr['short_target']
        sma = curr['sma']
        atr = curr['atr']
        
        action = "HOLD"
        entry_price = curr_close
        stop_loss = None
        reason = ""

        # Long 진입
        is_bull_trend = curr_close > sma
        
        if is_bull_trend and (curr_high >= long_target):
            action = "BUY"
            entry_price = max(curr_open, long_target)
            stop_loss = entry_price - (atr * self.sl_multiplier)
            reason = f"Long Breakout (Trailing)"

        # Short 진입
        elif (not is_bull_trend) and (curr_low <= short_target):
            action = "SELL"
            entry_price = min(curr_open, short_target)
            stop_loss = entry_price + (atr * self.sl_multiplier)
            reason = f"Short Breakout (Trailing)"
        
        return TradeSignal(
            timestamp=timestamp,
            action=action,
            entry_price=entry_price,
            stop_loss=stop_loss,
            reason=reason,
            # [핵심] 트레일링 스탑 정보를 백테스터에게 전달
            trailing_stop_multiplier=self.trailing_mult
        )