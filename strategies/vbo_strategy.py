import pandas as pd
import numpy as np
import ta
from lib.strategy import BaseStrategy

class VolatilityBreakoutStrategy(BaseStrategy):
    """
    [Optimized]
    변동성 돌파 전략
    - iloc 제거 및 NumPy Array 직접 접근 방식을 통한 고속화
    """
    def initialize(self):
        # 1. 파라미터 설정
        # [Fix] self.symbol 초기화가 누락되어 추가했습니다.
        self.symbol = self.params.get('symbol', 'BTC/USDT')
        
        self.k_window = self.params.get('k_window', 20)
        self.sma_period = self.params.get('sma_period', 50)
        self.sl_mult = self.params.get('sl_mult', 3.0)
        self.risk_pct = self.params.get('risk_per_trade', 0.2)
        self.leverage = self.params.get('leverage', 1.0)
        self.trailing_mult = self.params.get('trailing_mult', 2.0)
        self.atr_period = self.params.get('atr_period', 14)

        # 2. 데이터 준비 (DataFrame이 없으면 Feed에서 재구성해야 함)
        # Optimize 모드에서는 self.feed.df가 None일 수 있음 -> NumPy로 구성
        if self.feed.df is None:
            df = pd.DataFrame({
                'open': self.feed.opens,
                'high': self.feed.highs,
                'low': self.feed.lows,
                'close': self.feed.closes,
                'volume': self.feed.volumes
            }, index=self.feed.dates)
        else:
            df = self.feed.df.copy()

        # ----------------------------------------------------
        # 지표 계산 로직
        # ----------------------------------------------------
        df['date_str'] = df.index.normalize()
        
        # 일봉 리샘플링
        df_daily = df.resample('1D').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
        })
        
        # 지표 계산
        df_daily['range'] = df_daily['high'] - df_daily['low']
        noise = 1 - np.abs(df_daily['close'] - df_daily['open']) / (df_daily['range'] + 1e-9)
        df_daily['k'] = noise.rolling(window=self.k_window).mean()
        df_daily['sma'] = ta.trend.sma_indicator(df_daily['close'], window=self.sma_period)
        df_daily['atr'] = ta.volatility.average_true_range(
            df_daily['high'], df_daily['low'], df_daily['close'], window=self.atr_period
        )

        # Shift & Rename (전일 지표 사용)
        df_daily_shifted = df_daily.shift(1)
        rename_map = {
            'range': 'prev_range', 'k': 'prev_k', 'sma': 'prev_sma', 'atr': 'prev_atr'
        }
        df_daily_shifted.rename(columns=rename_map, inplace=True)
        
        # 병합 준비
        df_daily_final = df_daily_shifted[list(rename_map.values())].copy()
        df_daily_final['date_str'] = df_daily_final.index.normalize()
        
        # 당일 시가 (Daily Open) - Shift 없이 가져옴
        current_daily_opens = df_daily['open'].rename("daily_open").to_frame()
        current_daily_opens['date_str'] = current_daily_opens.index.normalize()

        # 1시간봉에 병합 (Merge)
        # reset_index()로 인덱스를 컬럼으로 내림
        df_merged = pd.merge(df.reset_index(), df_daily_final, on='date_str', how='left')
        df_merged = pd.merge(df_merged, current_daily_opens, on='date_str', how='left')
        
        # Target 계산
        df_merged['long_target'] = df_merged['daily_open'] + (df_merged['prev_range'] * df_merged['prev_k'])

        # ----------------------------------------------------
        # [핵심 최적화] 계산된 지표를 NumPy Array로 변환하여 멤버 변수에 저장
        # DataFrame 접근(iloc)을 피하기 위함
        # ----------------------------------------------------
        self.arr_long_target = df_merged['long_target'].values.astype('float32')
        self.arr_prev_sma = df_merged['prev_sma'].values.astype('float32')
        self.arr_prev_atr = df_merged['prev_atr'].values.astype('float32')
        
        # [Visualizer 지원] 단일 실행(main.py) 시에는 분석용 DataFrame을 저장해야 함
        if self.feed.df is not None:
            # merge로 인해 인덱스가 사라졌으므로 다시 복구
            if 'datetime' in df_merged.columns:
                df_merged.set_index('datetime', inplace=True)
            elif 'index' in df_merged.columns:
                df_merged.set_index('index', inplace=True)
            self.df = df_merged
        else:
            self.df = None # Optimize 모드에선 메모리 절약

        self.current_sl_price = 0.0

    def next(self):
        """
        [고속화된 Next 로직]
        DataFrame.iloc 대신 미리 캐싱한 NumPy Array 인덱싱 사용
        """
        idx = self.feed.current_idx
        
        # 배열 범위 체크 (Safety)
        if idx >= len(self.arr_long_target): return

        # 1. NumPy Array Access (매우 빠름)
        # NaN 체크 (np.isnan이 pd.isna보다 빠름)
        tgt_price = self.arr_long_target[idx]
        sma_val = self.arr_prev_sma[idx]
        atr_val = self.arr_prev_atr[idx]

        if np.isnan(tgt_price) or np.isnan(sma_val):
            return

        # 현재가 정보는 DataFeed가 이미 NumPy에서 꺼내서 들고 있음
        current_close = self.feed.close
        current_high = self.feed.high
        current_low = self.feed.low
        current_open = self.feed.open
        current_date = self.feed.date # datetime object

        pos = self.position 

        # ------------------------------------------
        # 진입 로직
        # ------------------------------------------
        if pos is None:
            # 추세 필터: 현재가 > 전일 SMA
            if current_close > sma_val:
                # 돌파 확인
                if current_high >= tgt_price:
                    entry_price = max(tgt_price, current_open)
                    
                    sl_dist = atr_val * self.sl_mult
                    sl_price = entry_price - sl_dist
                    
                    qty = self._calculate_entry_qty(entry_price, sl_dist)
                    
                    if qty > 0:
                        self.buy(quantity=qty, price=entry_price, symbol=self.symbol, immediate=True)
                        self.current_sl_price = sl_price
                        
                        # Intra-bar Loss Check
                        if current_low <= sl_price:
                             self.close(symbol=self.symbol, price=sl_price, immediate=True)

        # ------------------------------------------
        # 청산 로직
        # ------------------------------------------
        else:
            # 트레일링 스탑
            new_sl = current_high - (atr_val * self.trailing_mult)
            if new_sl > self.current_sl_price:
                self.current_sl_price = new_sl
            
            # Stop Loss (Intra-bar)
            if current_low <= self.current_sl_price:
                self.close(symbol=self.symbol, price=self.current_sl_price, immediate=True)
                return

            # Time Cut (날짜 변경 시 청산)
            # datetime.date() 비교는 Python 객체 비교라 약간 느리지만 필수 로직
            if pos.entry_time.date() < current_date.date():
                self.close(symbol=self.symbol, price=current_open, immediate=True)

    def _calculate_entry_qty(self, entry_price, sl_dist):
        """자금 관리"""
        balance = self.equity
        risk_amount = balance * self.risk_pct
        
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