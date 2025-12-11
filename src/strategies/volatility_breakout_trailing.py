import pandas as pd
import ta
import numpy as np
from typing import Dict, Any

from src.strategies.base import BaseStrategy, TradeSignal
from config.settings import Config

class VolatilityBreakoutTrailingStrategy(BaseStrategy):
    """
    [Step 3 Modified] 
    1시간봉(1h) 데이터를 받아 일봉(1d) 기준의 변동성 돌파 전략을 수행하는 클래스
    - Resampling을 통해 일봉 지표를 계산하고 Hourly 데이터에 매핑합니다.
    - Wide SL (넓은 손절) 전략이 반영됩니다.
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        
        # 설정 로드
        self.sma_period = self.config.get('sma_period', Config.SMA_PERIOD)
        self.k_window = self.config.get('k_window', Config.VBO_K_WINDOW)
        self.atr_period = self.config.get('atr_period', Config.ATR_PERIOD)
        self.sl_multiplier = self.config.get('sl_multiplier', Config.STOP_LOSS_MULTIPLIER)
        
        # 트레일링 스탑 설정
        if not Config.USE_TRAILING_STOP:
             self.trailing_mult = None
        else:
            default_trailing = Config.TRAILING_STOP_MULTIPLIER
            self.trailing_mult = self.config.get('trailing_mult', default_trailing)

    def calculate_indicators(self, df_hourly: pd.DataFrame) -> pd.DataFrame:
        """
        1시간봉 데이터를 받아 일봉 지표를 계산하고 병합합니다.
        """
        # 원본 데이터 보존
        df = df_hourly.copy()
        
        # 1. 일봉(Daily)으로 리샘플링 (하루의 시가, 고가, 저가, 종가 구하기)
        # 시간 기준은 00:00 ~ 23:59 가정 (바이낸스 UTC 기준)
        df_daily = df.resample('1D').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
        
        # 2. 일봉 기준 지표 계산
        # (주의: 여기서 계산되는 값들은 '오늘' 장이 끝나야 확정되는 값들임)
        df_daily['range'] = df_daily['high'] - df_daily['low']
        
        # Noise Ratio & K (일봉 기준)
        noise = 1 - np.abs(df_daily['close'] - df_daily['open']) / (df_daily['range'] + 1e-9)
        df_daily['k'] = noise.rolling(window=self.k_window).mean()
        
        # SMA & ATR (일봉 기준)
        df_daily['sma'] = ta.trend.sma_indicator(df_daily['close'], window=self.sma_period)
        df_daily['atr'] = ta.volatility.average_true_range(df_daily['high'], df_daily['low'], df_daily['close'], window=self.atr_period)

        # 3. [중요] 지표 Shift (하루 뒤로 미루기)
        # 오늘의 목표가는 '어제' 데이터를 기준으로 계산되어야 함.
        # 따라서 계산된 일봉 지표를 한 칸 밑으로(미래로) 내립니다.
        df_daily_shifted = df_daily.shift(1)
        
        # 컬럼 이름 변경 (혼동 방지)
        # 예: range -> prev_day_range
        rename_map = {
            'range': 'prev_range',
            'k': 'prev_k',
            'sma': 'prev_sma', # 어제 종가 기준 SMA
            'atr': 'prev_atr',
            'high': 'prev_day_high',
            'low': 'prev_day_low',
            'close': 'prev_day_close',
            'open': 'prev_day_open'
        }
        df_daily_shifted.rename(columns=rename_map, inplace=True)
        
        # 필요한 컬럼만 남기기
        cols_to_use = list(rename_map.values())
        df_daily_final = df_daily_shifted[cols_to_use]

        # 4. 1시간봉 데이터에 일봉 지표 매핑 (Merge/Join)
        # ffill(Forward Fill)을 사용하여, 1월 2일 00:00의 일봉 데이터를
        # 1월 2일 01:00, 02:00... 23:00까지 동일하게 적용합니다.
        
        # 인덱스(날짜)를 기준으로 병합하기 위해 날짜 컬럼 생성
        df['date_str'] = df.index.normalize() # 시분초 제거한 날짜
        df_daily_final['date_str'] = df_daily_final.index.normalize()
        
        # 병합
        df_merged = pd.merge(df.reset_index(), df_daily_final, on='date_str', how='left').set_index('datetime')
        
        # 5. 최종 목표가(Target) 계산 (매 시간마다 동일한 값)
        # 공식: 오늘 시가(일봉 시가가 아님! 변동성 돌파는 '당일 기준가' + Range * K)
        # 보통 변동성 돌파는 'Daily Open'을 기준으로 합니다.
        
        daily_opens = df.resample('1D')['open'].first()
        df_merged['daily_open'] = df_merged['date_str'].map(daily_opens)
        
        # 목표가 계산
        df_merged['long_target'] = df_merged['daily_open'] + (df_merged['prev_range'] * df_merged['prev_k'])
        
        # [Optimization] 6. 진입 신호 선계산 (Vectorized Signal Generation)
        # 루프 내에서 매번 계산하는 대신, 여기서 한 번에 계산합니다.
        # 조건 1: 추세 필터 (현재가 > 어제 SMA) - 보수적 관점: '시가'가 SMA 위에 있거나 '종가'가 위에 있거나. 
        # 여기서는 원본 로직 유지: 현재가(close) 기준이지만, 진입 시점(High가 쳤을 때)을 고려해야 함.
        # 하지만 High가 Target을 쳤다는 건 가격이 상승했다는 뜻이므로, Bull Trend일 확률이 높음.
        # 정확히는 "전일 종가 > 전일 SMA" 조건을 많이 씀. (코드상 is_bull_trend = curr_close > daily_sma 였음)
        # curr_close > daily_sma는 실시간 변동 조건이므로 벡터화 시 '종가' 기준으로 근사하거나,
        # 엄밀하게 하려면 '전일 종가' 기준으로 필터링하는 것이 일반적임 (Pre-filtering).
        # 기존 로직을 최대한 유지하기 위해 '현재 close > sma' 조건을 그대로 벡터화.
        
        condition_trend = df_merged['close'] > df_merged['prev_sma']
        condition_breakout = df_merged['high'] >= df_merged['long_target']
        
        # buy_signal 컬럼 추가 (Boolean)
        df_merged['buy_signal'] = condition_trend & condition_breakout
        
        # 데이터 정리
        df_merged.dropna(inplace=True)
        
        return df_merged

    def generate_signal(self, curr_row: Any) -> TradeSignal:
        """
        현재(1시간봉) 데이터를 보고 진입 여부 판단
        :param curr_row: itertuples()로 생성된 NamedTuple (Index, open, high, low, close, ...)
        """
        # itertuples()의 결과인 NamedTuple은 .Index로 인덱스(datetime)에 접근합니다.
        timestamp = curr_row.Index
        
        # 현재가 정보 (1시간봉)
        curr_close = curr_row.close
        curr_high = curr_row.high # 이번 시간의 고가
        
        # 일봉 기준 지표 (하루 종일 같은 값)
        long_target = curr_row.long_target
        daily_sma = curr_row.prev_sma # 어제까지의 추세
        daily_atr = curr_row.prev_atr
        
        action = "HOLD"
        entry_price = curr_close
        stop_loss = None
        reason = ""

        # --- 진입 판단 로직 ---
        # 1. 추세 필터: 어제 종가가 어제 SMA보다 높았는가? (혹은 현재가가 SMA보다 높은가)
        is_bull_trend = curr_close > daily_sma
        
        # 2. 돌파 확인
        # 이번 1시간 캔들의 고가(High)가 목표가를 건드렸는가?
        if is_bull_trend and (curr_high >= long_target):
            action = "BUY"
            
            # 진입가: 목표가와 현재 시가(1h open) 중 큰 값 (Gap 보정)
            # 보통 변동성 돌파는 '목표가' 체결 가정
            entry_price = long_target
            
            # 손절가 설정 (Wide SL)
            stop_loss = entry_price - (daily_atr * self.sl_multiplier)
            
            reason = f"Hit Target {long_target:.2f} (DailyATR: {daily_atr:.2f})"

        return TradeSignal(
            timestamp=timestamp,
            action=action,
            entry_price=entry_price,
            stop_loss=stop_loss,
            reason=reason,
            trailing_stop_multiplier=self.trailing_mult
        )