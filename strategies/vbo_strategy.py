import pandas as pd
import numpy as np
import ta
from lib.strategy import BaseStrategy
from lib.objects import OrderSide

class VolatilityBreakoutStrategy(BaseStrategy):
    """
    변동성 돌파 전략 (Volatility Breakout) - Daily Logic on Hourly Data
    [Refactored to match test_deep logic]
    1. 1시간봉 데이터를 받아 일봉(Daily) 지표로 변환하여 계산
    2. 진입: 현재가가 (당일 시가 + 전일 변동폭 * K) 돌파 시
    3. 필터: 현재가가 전일 SMA보다 높을 때 (추세장)
    """
    def initialize(self):
        # 1. 파라미터 설정
        self.symbol = self.params.get('symbol', 'BTC/USDT')
        self.k_window = self.params.get('k_window', 20)
        self.sma_period = self.params.get('sma_period', 50)
        self.sl_mult = self.params.get('sl_mult', 3.0)
        self.risk_pct = self.params.get('risk_per_trade', 0.2)
        self.leverage = self.params.get('leverage', 1.0)
        self.trailing_mult = self.params.get('trailing_mult', 2.0)
        self.atr_period = self.params.get('atr_period', 14)

        # 2. 데이터 가공 (Hourly -> Daily Indicators)
        df = self.feed.df.copy() # 원본 보존을 위해 copy
        
        # (1) 날짜 매핑을 위한 컬럼
        df['date_str'] = df.index.normalize()

        # (2) 일봉(Daily)으로 리샘플링
        # 시간 기준은 데이터의 인덱스(DatetimeIndex) 사용
        df_daily = df.resample('1D').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
        
        # (3) 일봉 기준 지표 계산
        # Range & Noise
        df_daily['range'] = df_daily['high'] - df_daily['low']
        noise = 1 - np.abs(df_daily['close'] - df_daily['open']) / (df_daily['range'] + 1e-9)
        df_daily['k'] = noise.rolling(window=self.k_window).mean()
        
        # SMA & ATR (일봉 기준)
        df_daily['sma'] = ta.trend.sma_indicator(df_daily['close'], window=self.sma_period)
        
        # ATR 계산 (ta 라이브러리 활용)
        df_daily['atr'] = ta.volatility.average_true_range(
            df_daily['high'], df_daily['low'], df_daily['close'], window=self.atr_period
        )

        # (4) 지표 Shift (하루 뒤로 미루기)
        # 오늘의 타겟은 '어제' 지표를 사용함
        df_daily_shifted = df_daily.shift(1)
        
        # 컬럼명 변경 (Hourly 데이터와 섞이지 않게)
        rename_map = {
            'range': 'prev_range',
            'k': 'prev_k',
            'sma': 'prev_sma',
            'atr': 'prev_atr',
            'open': 'prev_open',
            'close': 'prev_close',
            'high': 'prev_high',
            'low': 'prev_low'
        }
        df_daily_shifted.rename(columns=rename_map, inplace=True)
        
        # 필요한 컬럼만 선택
        daily_cols = list(rename_map.values())
        df_daily_final = df_daily_shifted[daily_cols]
        df_daily_final['date_str'] = df_daily_final.index.normalize()

        # (5) 1시간봉 데이터에 일봉 지표 병합
        # index가 사라지지 않도록 reset_index 후 병합
        df_merged = pd.merge(df.reset_index(), df_daily_final, on='date_str', how='left')
        
        # 병합 후 다시 인덱스 설정 (원래 인덱스 이름이 datetime인지 확인 필요하지만 보통 0번 컬럼)
        # reset_index()를 하면 원래 인덱스가 컬럼으로 내려옴. 이름이 'datetime'이거나 'index'일 것임.
        if 'datetime' in df_merged.columns:
            df_merged.set_index('datetime', inplace=True)
        elif 'index' in df_merged.columns:
            df_merged.set_index('index', inplace=True)

        # (6) 당일 시가(Daily Open) 매핑
        # df_merged['daily_open']은 '오늘'의 시가여야 함.
        # 위에서 resample('1D')['open'] 한 것을 shift 하지 않고 가져와야 함.
        current_daily_opens = df_daily['open'].rename("daily_open")
        current_daily_opens = current_daily_opens.to_frame()
        current_daily_opens['date_str'] = current_daily_opens.index.normalize()
        
        df_merged = pd.merge(df_merged.reset_index(), current_daily_opens, on='date_str', how='left')
        
        if 'datetime' in df_merged.columns:
            df_merged.set_index('datetime', inplace=True)
        elif 'index' in df_merged.columns:
            df_merged.set_index('index', inplace=True)

        # (7) 최종 타겟 가격 계산
        # Target = Daily Open + (Prev Range * Prev K)
        df_merged['long_target'] = df_merged['daily_open'] + (df_merged['prev_range'] * df_merged['prev_k'])
        
        # (8) 계산된 데이터프레임을 self.df로 설정
        self.df = df_merged
        
        # 3. 런타임 변수
        self.current_sl_price = 0.0

    def next(self):
        """매 시간(Bar)마다 호출되는 로직"""
        idx = self.feed.current_idx
        
        # 초기 데이터 부족 시 패스
        if idx >= len(self.df): return # Safety
        
        # iloc 접근
        current = self.df.iloc[idx]
        
        # 지표가 NaN이면(초기 구간) 패스
        if pd.isna(current['long_target']) or pd.isna(current['prev_sma']):
            return

        pos = self.position 

        # ==========================================
        # [전략 1] 포지션이 없을 때 (진입 판단)
        # ==========================================
        if pos is None:
            # 추세 필터: 현재가가 전일 SMA보다 높은가?
            is_uptrend = current['close'] > current['prev_sma']
            breakout_price = current['long_target']
            
            # 돌파 확인: 고가가 타겟 이상인가
            if is_uptrend and current['high'] >= breakout_price:
                # 진입가: 목표가와 현재 시가(1h open) 중 큰 값 (Gap 보정)
                # 보통 변동성 돌파는 '목표가' 체결 가정, 단 갭상승 시 시가 체결
                # 주의: current['open']은 1시간봉 시가임
                entry_price = max(breakout_price, current['open'])
                
                # 손절가 계산 (전일 ATR 기준)
                sl_dist = current['prev_atr'] * self.sl_mult
                sl_price = entry_price - sl_dist
                
                qty = self._calculate_entry_qty(entry_price, sl_dist)
                
                if qty > 0:
                    self.buy(quantity=qty, price=entry_price, symbol=self.symbol, immediate=True)
                    self.current_sl_price = sl_price
                    
                    # Intra-bar Loss Check (진입한 봉에서 바로 손절가를 건드렸는지 확인)
                    if current['low'] <= sl_price:
                         self.close(symbol=self.symbol, price=sl_price, immediate=True)

        # ==========================================
        # [전략 2] 포지션 보유 중 (청산/관리)
        # ==========================================
        else:
            # 1. 트레일링 스탑 업데이트
            # 전일 ATR 사용
            new_sl = current['high'] - (current['prev_atr'] * self.trailing_mult)
            if new_sl > self.current_sl_price:
                self.current_sl_price = new_sl
            
            # 2. 청산 실행 (Intra-bar Logic)
            if current['low'] <= self.current_sl_price:
                self.close(symbol=self.symbol, price=self.current_sl_price, immediate=True)
                return

            # 3. 타임 컷 (Time Cut) - 날짜가 바뀌면 청산
            # 진입 날짜 < 현재 날짜 이면 청산
            # current.name은 DatetimeIndex의 값 (Timestamp)
            if pos.entry_time.date() < current.name.date():
                self.close(symbol=self.symbol, price=current['open'], immediate=True)

    def _calculate_entry_qty(self, entry_price, sl_dist):
        """자금 관리: 손절폭 대비 수량 계산"""
        balance = self.equity
        risk_amount = balance * self.risk_pct
        
        # 격리 마진 안전장치: 청산가보다 손절가가 아래면 안됨
        stop_price = entry_price - sl_dist
        liq_price = entry_price * (1 - (1/self.leverage) + 0.005)
        
        if stop_price <= liq_price:
            stop_price = liq_price * 1.002
            sl_dist = entry_price - stop_price
        
        if sl_dist <= 0: return 0
        qty = risk_amount / sl_dist
        
        max_qty = (balance * self.leverage) / entry_price
        qty = min(qty, max_qty)
        
        return qty