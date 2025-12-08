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
        # 따라서 1시간봉의 open이 아니라, 위에서 가져온 'prev_day_close'(혹은 오늘 날짜의 시가)를 써야 합니다.
        # 바이낸스 기준 00:00의 Open price가 그날의 기준가가 됩니다.
        
        # 병합된 데이터에는 '오늘의 시가' 정보가 없으므로(shift 했으니까), 
        # 다시 df_daily에서 '오늘 시가'를 가져와야 합니다. 
        # 하지만 간단하게: df_merged['open']은 매 시간의 시가이므로 쓰면 안됨.
        
        # [수정 로직] Daily Open 구하기
        # 현재 캔들이 속한 날짜의 00:00 시가 찾기
        daily_opens = df.resample('1D')['open'].first()
        # 이것도 시간 단위로 매핑
        df_merged['daily_open'] = df_merged['date_str'].map(daily_opens)
        
        # 목표가 계산
        # Target = 당일 시가 + (전일 Range * 전일 K)
        df_merged['long_target'] = df_merged['daily_open'] + (df_merged['prev_range'] * df_merged['prev_k'])
        
        # 데이터 정리
        df_merged.dropna(inplace=True)
        
        return df_merged

    def generate_signal(self, current_slice: pd.DataFrame) -> TradeSignal:
        """
        현재(1시간봉) 데이터를 보고 진입 여부 판단
        """
        if len(current_slice) < 1:
            return TradeSignal(timestamp=None, action="HOLD", entry_price=0)

        curr = current_slice.iloc[-1]
        timestamp = curr.name
        
        # 현재가 정보 (1시간봉)
        curr_close = curr['close']
        curr_high = curr['high'] # 이번 시간의 고가
        curr_low = curr['low']   # 이번 시간의 저가
        
        # 일봉 기준 지표 (하루 종일 같은 값)
        long_target = curr['long_target']
        daily_sma = curr['prev_sma'] # 어제까지의 추세
        daily_atr = curr['prev_atr']
        
        action = "HOLD"
        entry_price = curr_close
        stop_loss = None
        reason = ""

        # --- 진입 판단 로직 ---
        # 1. 추세 필터: 어제 종가가 어제 SMA보다 높았는가? (혹은 현재가가 SMA보다 높은가)
        # 보통 래리 윌리엄스 전략은 '현재 가격'이 이동평균선 위에 있을 때를 선호
        is_bull_trend = curr_close > daily_sma
        
        # 2. 돌파 확인
        # 이번 1시간 캔들의 고가(High)가 목표가를 건드렸는가?
        if is_bull_trend and (curr_high >= long_target):
            action = "BUY"
            
            # 진입가: 목표가와 현재 시가(1h open) 중 큰 값 (Gap 보정)
            # 하지만 백테스트에서는 '목표가'에 체결되었다고 가정하는 게 일반적 (Slippage 별도)
            entry_price = long_target
            
            # 손절가 설정 (Wide SL)
            # 진입가 - (일봉 ATR * Multiplier)
            # 예: Multiplier가 4.0이면 ATR의 4배만큼 여유를 둠
            stop_loss = entry_price - (daily_atr * self.sl_multiplier)
            
            reason = f"Hit Target {long_target:.2f} (DailyATR: {daily_atr:.2f})"

        # 숏(Sell) 로직은 현재 롱 전용 포트폴리오(BTC/ETH)에 집중하기 위해 생략하거나
        # 필요 시 대칭적으로 구현 (Short Target = Daily Open - Range * K)
        # 여기서는 Long Only 로직만 명확히 기술
        
        return TradeSignal(
            timestamp=timestamp,
            action=action,
            entry_price=entry_price,
            stop_loss=stop_loss,
            reason=reason,
            trailing_stop_multiplier=self.trailing_mult
        )