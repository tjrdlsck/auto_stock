import sys
import os
import json
import joblib
import pandas as pd
import pandas_ta as ta
import numpy as np
import ccxt.async_support as ccxt
from datetime import datetime
import sqlite3
import asyncio
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type



# 프로젝트 경로 설정
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import CONFIG, MODELS_DIR, DATA_DIR

# 설정 로드
load_dotenv()
IS_TESTNET = os.getenv("TRADING_MODE_TESTNET") == 'True'

class BinanceTrader:
    def __init__(self, messenger_callback=None):
        """
        :param messenger_callback: 봇에게 메시지를 보낼 때 사용하는 함수
        """
        self.messenger = messenger_callback
        
        # 거래소 연결
        self.exchange = ccxt.binance({
            'apiKey': os.getenv("BINANCE_API_KEY"),
            'secret': os.getenv("BINANCE_SECRET_KEY"),
            'options': {'defaultType': 'future'},
            'enableRateLimit': True
        })
        
        if IS_TESTNET:
            self.exchange.set_sandbox_mode(True)
        
        # DB 경로 설정 및 초기화
        self.db_path = os.path.join(DATA_DIR, 'trade_state.db')
        self._init_db() # 데이터베이스 초기화 및 테이블 생성
        
        # [NEW] 매매 모드 설정 (DB에서 불러오기)
        self.mode = self._get_mode_from_db()

        # 전략 파라미터 (V8)
        self.RSI_BUY = 55
        self.RSI_SELL = 45
        self.EMA_PERIOD = 20
        self.STOP_LOSS_ATR_MULTIPLIER = 2.0 # ATR 기반 손절 배수
        self.BREAKEVEN_PNL_PCT = 0.015      # 본절 전환 수익률 (+1.5%)
        self.COOLDOWN_BARS = 3              # 진입 쿨타임 (시간)
        self.TRAIL_PCT = 0.03               # 트레일링 스탑
        
        self.trade_lock = asyncio.Lock() # 동시성 제어를 위한 락
        self.cooldowns = {} # [NEW] 쿨다운 관리
        
        # 상태 관리 (SQLite로 대체)
        self.state = self._load_positions_from_db() # DB에서 초기 상태 로드

    # --- SQLite Helper Methods ---
    def _init_db(self):
        """데이터베이스를 초기화하고 필요한 테이블 생성 및 마이그레이션을 수행합니다."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 1. positions 테이블 생성
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS positions (
                symbol TEXT PRIMARY KEY,
                entry_price REAL,
                highest_price REAL,
                lowest_price REAL,
                side TEXT,
                amount REAL
            )
        ''')
        
        # 2. system_settings 테이블 생성
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_settings (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        ''')
        
        # --- [마이그레이션] positions 테이블에 entry_regime 컬럼 추가 ---
        cursor.execute("PRAGMA table_info(positions)")
        columns = [info[1] for info in cursor.fetchall()]
        if 'entry_regime' not in columns:
            print("⏳ [DB 마이그레이션] 'positions' 테이블에 'entry_regime' 컬럼 추가 중...")
            cursor.execute("ALTER TABLE positions ADD COLUMN entry_regime TEXT")
        
        # --- [마이그레이션] positions 테이블에 stop_loss_price 컬럼 추가 ---
        if 'stop_loss_price' not in columns:
            print("⏳ [DB 마이그레이션] 'positions' 테이블에 'stop_loss_price' 컬럼 추가 중...")
            cursor.execute("ALTER TABLE positions ADD COLUMN stop_loss_price REAL DEFAULT 0.0")

        # --- 기본값 설정 ---
        cursor.execute("INSERT OR IGNORE INTO system_settings (key, value) VALUES ('mode', 'OFF')")
        
        conn.commit()
        conn.close()
        print(f"✅ SQLite DB 초기화 및 마이그레이션 완료: {self.db_path}")

    def _get_mode_from_db(self):
        """DB에서 현재 매매 모드를 불러옵니다."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT value FROM system_settings WHERE key='mode'")
        result = cursor.fetchone()
        conn.close()
        return result[0] if result else 'OFF'

    def _load_positions_from_db(self):
        """데이터베이스에서 현재 포지션 상태를 로드합니다."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT symbol, entry_price, highest_price, lowest_price, side, amount, entry_regime, stop_loss_price FROM positions")
        rows = cursor.fetchall()
        conn.close()
        
        state = {}
        for row in rows:
            symbol, entry_price, highest_price, lowest_price, side, amount, entry_regime, stop_loss_price = row
            state[symbol] = {
                'entry_price': entry_price,
                'high': highest_price,
                'low': lowest_price,
                'side': side,
                'amount': amount,
                'entry_regime': entry_regime,
                'stop_loss_price': stop_loss_price
            }
        return state

    def _upsert_position_to_db(self, symbol, entry_price, highest_price, lowest_price, side, amount, entry_regime, stop_loss_price):
        """단일 포지션을 데이터베이스에 저장하거나 업데이트합니다."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO positions (symbol, entry_price, highest_price, lowest_price, side, amount, entry_regime, stop_loss_price)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (symbol, entry_price, highest_price, lowest_price, side, amount, entry_regime, stop_loss_price))
        conn.commit()
        conn.close()

    def _delete_position_from_db(self, symbol):
        """데이터베이스에서 특정 포지션을 삭제합니다."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM positions WHERE symbol = ?", (symbol,))
        conn.commit()
        conn.close()
    # --- End SQLite Helper Methods ---

    def set_mode(self, new_mode):
        """모드를 변경하고 DB에 저장"""
        new_mode = new_mode.upper()
        if new_mode not in ['OFF', 'REAL', 'PAPER']:
            return "❌ 잘못된 모드입니다."
        
        self.mode = new_mode
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("UPDATE system_settings SET value = ? WHERE key = 'mode'", (new_mode,))
        conn.commit()
        conn.close()
        return f"✅ 시스템 모드가 **{new_mode}**로 변경 및 저장되었습니다."

    def log(self, msg):
        print(msg)
        # 메신저 콜백이 있으면 디스코드로도 전송 (중요 메시지만)
        if self.messenger and "✅" in msg: 
            # execute_order에서 따로 처리하므로 여기선 단순 로그만
            pass

    def load_state(self):
        if os.path.exists(self.STATE_FILE):
            try:
                with open(self.STATE_FILE, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def save_state(self):
        with open(self.STATE_FILE, 'w') as f:
            json.dump(self.state, f, indent=4)

    def get_risk_setting(self, symbol):
        if 'BTC' in symbol or 'ETH' in symbol:
            return {'leverage': 2, 'risk_pct': 0.95}
        elif 'SOL' in symbol:
            return {'leverage': 1, 'risk_pct': 0.80}
        else:
            return {'leverage': 1, 'risk_pct': 0.50}

    def calculate_position_size(self, usdt_balance, atr_value, target_risk=0.02):
        """
        ATR 기반으로 포지션 사이즈(수량) 계산
        :param usdt_balance: 사용 가능한 USDT 잔고
        :param atr_value: 현재 ATR 값
        :param target_risk: 한 번의 거래에서 감수할 최대 손실 비율 (예: 2%)
        :return: 계산된 포지션 수량
        """
        # 손절 라인을 ATR의 2배로 설정
        stop_loss_dist_in_price = atr_value * 2
        
        # 분모가 0이 되는 것을 방지
        if stop_loss_dist_in_price == 0:
            return 0
            
        # 리스크 관리 공식: (총 자본 * 리스크%) / (손절 거리)
        # 즉, 손절이 나가도 총 자본의 2%만 잃도록 수량 조절
        risk_per_coin = stop_loss_dist_in_price
        qty = (usdt_balance * target_risk) / risk_per_coin
        
        return qty

    async def set_leverage(self, symbol, leverage):
        try:
            await self.exchange.set_leverage(leverage, symbol)
        except Exception as e:
            print(f"❌ 레버리지 설정 실패 ({symbol}, {leverage}): {e}")

    async def fetch_data_and_features(self, symbol):
        try:
            timeframe = CONFIG['TIMEFRAME']
            candles = await self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=CONFIG['CANDLE_LIMIT'])

            if not candles:
                print(f"❌ [{symbol}] 데이터 수집 실패. 캔들이 없습니다.")
                return None

            # --- 캔들 완성 여부 체크 ---
            last_candle_ts = pd.to_datetime(candles[-1][0], unit='ms', utc=True)
            timeframe_duration = pd.to_timedelta(timeframe)
            now_utc = datetime.now(last_candle_ts.tz) # 캔들 타임존과 동일한 시간대 사용

            # 마지막 캔들이 아직 닫히지 않았으면, 그 전 캔들까지만 사용
            if now_utc < last_candle_ts + timeframe_duration:
                print(f"🕯️ [{symbol}] 마지막 캔들 미완성. 분석에서 제외합니다. (현재: {now_utc}, 캔들 마감: {last_candle_ts + timeframe_duration})")
                candles = candles[:-1]
            
            if not candles:
                print(f"❌ [{symbol}] 사용할 완성된 캔들이 없습니다.")
                return None
            # --- ---------------- ---

            df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df.set_index('timestamp', inplace=True)

            # Feature Engineering (RSI_14가 기본값)
            df['Log_Returns'] = np.log(df['close'] / df['close'].shift(1))
            df['Range_Vol'] = (df['high'] - df['low']) / df['close']
            df.ta.rsi(length=14, append=True)
            df.ta.ema(length=self.EMA_PERIOD, append=True)
            df.ta.bbands(length=20, std=2, append=True) # Add Bollinger Bands
            df.ta.atr(append=True) # Add ATR
            df.ta.obv(append=True) # Add On-Balance Volume

            # Scaled Features
            window = 30
            features_to_scale = ['Log_Returns', 'Range_Vol', 'RSI_14', 'OBV']
            for col in features_to_scale:
                rolling_mean = df[col].rolling(window=window).mean()
                rolling_std = df[col].rolling(window=window).std()
                df[f'{col}_Scaled'] = (df[col] - rolling_mean) / (rolling_std + 1e-8)
            
            return df.dropna()
        except Exception as e:
            print(f"❌ 데이터 처리 중 에러 ({symbol}): {e}")
            return None

    def get_hmm_regime(self, df, symbol):
        try:
            clean_symbol = symbol.replace('/', '')
            model_path = os.path.join(MODELS_DIR, f"hmm_{clean_symbol}.pkl")
            
            if not os.path.exists(model_path): return None, None

            model = joblib.load(model_path)
            X = df[['Log_Returns_Scaled', 'Range_Vol_Scaled', 'RSI_14_Scaled', 'OBV_Scaled']].values
            hidden_states = model.predict(X)
            
            stats = pd.DataFrame(X, columns=['Ret', 'Vol', 'RSI', 'OBV'])
            stats['Regime'] = hidden_states
            regime_means = stats.groupby('Regime')['Ret'].mean()
            
            bear_id = regime_means.idxmin()
            bull_id = regime_means.idxmax()
            
            # Find the sideways ID
            all_ids = set(range(model.n_components))
            sideways_id = list(all_ids - {bull_id, bear_id})[0] if len(all_ids) > 2 else None
            
            return hidden_states[-1], {'bull': bull_id, 'bear': bear_id, 'sideways': sideways_id}
        except:
            return None, None

    # 네트워크 에러나 타임아웃 발생 시 1초 간격으로 최대 3번 재시도
    @retry(
        stop=stop_after_attempt(3), 
        wait=wait_fixed(1), 
        retry=retry_if_exception_type((ccxt.NetworkError, ccxt.RequestTimeout, ccxt.ExchangeError))
    )
    async def _create_market_order_with_retry(self, symbol, side, amount, params):
        """재시도 로직이 적용된 실제 시장가 주문 실행 함수"""
        return await self.exchange.create_market_order(symbol, side, amount, params)

    @retry(
        stop=stop_after_attempt(3), 
        wait=wait_fixed(1), 
        retry=retry_if_exception_type((ccxt.NetworkError, ccxt.RequestTimeout, ccxt.ExchangeError))
    )
    async def _create_limit_order_with_retry(self, symbol, side, amount, price, params):
        """재시도 로직이 적용된 실제 지정가 주문 실행 함수"""
        return await self.exchange.create_limit_order(symbol, side, amount, price, params)

    async def _smart_execute_order(self, symbol, side, amount, reduce_only=False):
        """
        지정가 주문을 먼저 시도하고, 일정 시간 미체결 시 시장가 주문으로 전환하는 스마트 실행 로직
        """
        params = {'reduceOnly': True} if reduce_only else {}
        timeout = CONFIG['LIMIT_ORDER_TIMEOUT_SEC']
        force_market = CONFIG['FORCE_MARKET_ORDER_ON_TIMEOUT']

        try:
            # 1. 최우선 호가 가져오기
            orderbook = await self.exchange.fetch_order_book(symbol)
            if side == 'buy':
                limit_price = orderbook['asks'][0][0] # 최우선 매도호가
            else: # sell
                limit_price = orderbook['bids'][0][0] # 최우선 매수호가
            
            # 2. 지정가 주문 시도
            limit_order = await self._create_limit_order_with_retry(symbol, side, amount, limit_price, params)
            order_id = limit_order['id']
            print(f"✅ [{symbol}] 지정가 주문 제출 완료. ID: {order_id}, 가격: {limit_price}, 수량: {amount}")

            # 3. 일정 시간 대기하며 체결 여부 확인
            start_time = asyncio.get_event_loop().time()
            while asyncio.get_event_loop().time() - start_time < timeout:
                order_status = await self.exchange.fetch_order(order_id, symbol)
                if order_status['status'] == 'closed':
                    print(f"🚀 [{symbol}] 지정가 주문 체결 완료. 가격: {order_status['average']}")
                    return order_status['average']
                if order_status['filled'] > 0: # 부분 체결
                    print(f"Partial fill for {symbol}. Filled: {order_status['filled']}")
                    # 이 경우, 나머지 물량을 시장가로 처리하거나 다시 지정가를 낼 수 있음.
                    # 여기서는 간단히 부분 체결된 가격을 반환하고 나머지는 무시 (실제 봇에서는 더 복잡한 로직 필요)
                    return order_status['average']

                await asyncio.sleep(1) # 1초마다 상태 확인

            # 4. 타임아웃 발생: 미체결 처리
            order_status = await self.exchange.fetch_order(order_id, symbol) # 최종 상태 확인
            if order_status['status'] != 'closed' and order_status['filled'] == 0:
                # 주문 취소
                await self.exchange.cancel_order(order_id, symbol)
                print(f"⚠️ [{symbol}] 지정가 주문 미체결로 취소됨. ID: {order_id}")

                if force_market:
                    print(f"🔥 [{symbol}] 시장가 주문으로 전환. 수량: {amount}")
                    market_order = await self._create_market_order_with_retry(symbol, side, amount, params)
                    print(f"🚀 [{symbol}] 시장가 주문 체결 완료. 가격: {market_order['average']}")
                    return market_order['average']
                else:
                    print(f"❌ [{symbol}] 지정가 미체결 후 시장가 전환 설정 OFF. 주문 취소만 진행.")
                    return None
            
            elif order_status['filled'] > 0 and order_status['status'] != 'closed': # 부분 체결 후 타임아웃
                print(f"⚠️ [{symbol}] 지정가 부분 체결 후 타임아웃. Filled: {order_status['filled']}, Remaining: {order_status['remaining']}")
                if force_market and order_status['remaining'] > 0:
                    # 잔여 물량 시장가 처리
                    print(f"🔥 [{symbol}] 잔여 물량 시장가 주문 전환. 수량: {order_status['remaining']}")
                    market_order = await self._create_market_order_with_retry(symbol, side, order_status['remaining'], params)
                    print(f"🚀 [{symbol}] 잔여 물량 시장가 체결 완료. 가격: {market_order['average']}")
                    return market_order['average'] # 부분 체결 + 시장가 체결의 평균 가격 반환 로직 필요. 단순화.
                else:
                    return order_status['average'] if order_status['filled'] > 0 else None

            return None # 어떤 경우에도 체결되지 않았을 때

        except Exception as e:
            print(f"❌ 스마트 주문 실행 중 에러 ({symbol}): {e}")
            import traceback
            traceback.print_exc()
            return None

    async def execute_order(self, symbol, side, amount, reduce_only=False):
        """모드에 따라 주문 실행 여부 결정"""
        
        # 1. 정지 상태면 즉시 리턴
        if self.mode == 'OFF':
            return None

        # 가격 정보 가져오기 (모의매매에서도 가격은 필요함)
        try:
            ticker = await self.exchange.fetch_ticker(symbol)
            current_price = ticker['last']
        except Exception as e:
            print(f"❌ 가격 조회 실패 ({symbol}): {e}")
            return None

        msg_type = "청산" if reduce_only else "진입"

        # 2. 모의매매 (PAPER)
        if self.mode == 'PAPER':
            # Apply slippage for a more realistic paper trading simulation
            if side == 'buy':
                current_price *= (1 + CONFIG['SLIPPAGE_PCT'])
            elif side == 'sell':
                current_price *= (1 - CONFIG['SLIPPAGE_PCT'])
            
            log_msg = f"🧪 [모의매매] {symbol} {side.upper()} {msg_type} 시그널! (가격: {current_price}, 수량: {amount:.4f})"
            print(log_msg)
            if self.messenger:
                # 모의매매용 별도 알림 함수 호출 or 기존 함수에 플래그 전달
                self.messenger(symbol, side, current_price, amount, f"[모의] {msg_type}")
            return current_price # 가상 체결 가격 반환

        # 3. 실매매 (REAL)
        if self.mode == 'REAL':
            exec_price = await self._smart_execute_order(symbol, side, amount, reduce_only)
            if exec_price:
                print(f"🚀 [실매매] {symbol} {side} 체결 완료 (가격: {exec_price})")
                if self.messenger:
                    self.messenger(symbol, side, exec_price, amount, msg_type)
                return exec_price
            else:
                error_msg = f"❌ [CRITICAL] 주문 최종 실패 ({symbol} {side} {amount:.4f}): 스마트 실행 실패"
                print(error_msg)
                if self.messenger:
                    self.messenger(symbol, side, 0, amount, f"[긴급] 주문 실패", mention_everyone=True)
                return None

    # ----------------------------------------------
    # [NEW] 디스코드 봇 보고용 함수
    # ----------------------------------------------
    async def get_balance(self):
        """현재 USDT 잔고 조회"""
        try:
            balance = await self.exchange.fetch_balance()
            return balance['free']['USDT'], balance['total']['USDT']
        except Exception as e:
            print(f"❌ 잔고 조회 실패: {e}")
            return 0.0, 0.0

    async def get_positions(self):
        """현재 활성화된 포지션 조회"""
        active_positions = []
        try:
            # 바이낸스는 모든 심볼 포지션을 줌, 필터링 필요
            positions = await self.exchange.fetch_positions()
            for p in positions:
                if float(p['contracts']) > 0:
                    active_positions.append({
                        'symbol': p['symbol'],
                        'side': p['side'],
                        'amount': float(p['contracts']),
                        'entryPrice': float(p['entryPrice']),
                        'unrealizedPnl': float(p['unrealizedPnl']),
                        'leverage': p['leverage']
                    })
            return active_positions
        except Exception as e:
            print(f"❌ 포지션 조회 실패: {e}")
            return []

    async def force_close(self, symbol):
        """디스코드 명령어로 특정 심볼 강제 청산"""
        try:
            positions_data = await self.exchange.fetch_positions([symbol])
            positions = [p for p in positions_data if p['symbol'] == symbol]
            if not positions:
                return f"⚠️ {symbol} 활성화된 포지션이 없습니다."
            
            pos_amt = float(positions[0]['contracts'])
            pos_side = positions[0]['side']
            
            if pos_amt == 0:
                return f"⚠️ {symbol} 포지션 수량이 0입니다."

            # 반대 주문으로 청산
            side = 'sell' if pos_side == 'long' else 'buy'
            price = await self.execute_order(symbol, side, pos_amt, reduce_only=True)
            
            if price:
                # 상태 파일에서 삭제
                if symbol in self.state: del self.state[symbol] # Keep for in-memory state consistency
                self._delete_position_from_db(symbol) # Call new method to remove from DB
                return f"✅ **{symbol}** 강제 청산 완료! (가격: {price})"
            else:
                return f"❌ {symbol} 청산 주문 실패."
                
        except Exception as e:
            return f"❌ 강제 청산 중 에러: {e}"


    async def safe_force_close(self, symbol):
        """🚨 안전한 강제 청산 (주문 취소 -> 청산 -> DB삭제)"""
        async with self.trade_lock: # 락을 걸어서 스케줄러 간섭 차단
            print(f"🔒 [{symbol}] 강제 청산 작업 시작 (Lock 획득)")
            # 1. 미체결 주문 취소
            try:
                await self.exchange.cancel_all_orders(symbol)
                print(f"🧹 [{symbol}] 미체결 주문 일괄 취소 완료")
            except Exception as e:
                print(f"⚠️ [{symbol}] 미체결 주문 취소 중 오류 (무시): {e}")

            # 2. 포지션 청산 (기존 로직 재활용)
            res = await self.force_close(symbol)
            
            print(f"🔓 [{symbol}] 강제 청산 작업 완료 (Lock 해제)")
            return res

    # ----------------------------------------------
    # 메인 로직 (외부에서 호출)
    # ----------------------------------------------
    async def run_logic(self):
        async with self.trade_lock:
            if self.mode == 'OFF':
                return "⏸️ 봇이 정지 상태입니다."
            
            # [NEW] 쿨다운 카운터 감소
            for symbol in list(self.cooldowns.keys()):
                self.cooldowns[symbol] -= 1
                if self.cooldowns[symbol] <= 0:
                    del self.cooldowns[symbol]

            try:
                balance = await self.exchange.fetch_balance()
                usdt_balance = balance['free']['USDT']
            except Exception as e:
                return f"❌ 잔고 조회 실패: {e}"

            self.state = self._load_positions_from_db()
            
            for symbol in CONFIG['SYMBOLS']:
                df = await self.fetch_data_and_features(symbol)
                if df is None: continue
                
                curr_price = df['close'].iloc[-1]
                curr_rsi = df['RSI_14'].iloc[-1]
                curr_ema = df[f'EMA_{self.EMA_PERIOD}'].iloc[-1]
                curr_atr = df['ATRr_14'].iloc[-1]

                regime, r_map = self.get_hmm_regime(df, symbol)
                if regime is None: continue
                
                positions_raw = await self.exchange.fetch_positions([symbol])
                positions = [p for p in positions_raw if p['symbol'] == symbol and float(p['contracts']) > 0]
                pos_amt = float(positions[0]['contracts']) if positions else 0.0
                pos_side = positions[0]['side'] if positions else None
                has_position = pos_amt > 0
                
                risk_setting = self.get_risk_setting(symbol)
                leverage = risk_setting['leverage']
                
                # --- 1. 청산 로직 (V8) ---
                if has_position:
                    state = self.state.get(symbol, {})
                    entry_price = state.get('entry_price', 0)
                    if not entry_price: continue # Should not happen

                    pnl_pct = (curr_price - entry_price) / entry_price if pos_side == 'long' else (entry_price - curr_price) / entry_price
                    stop_loss_price = state.get('stop_loss_price', 0)

                    # 본절 로직
                    if stop_loss_price != entry_price and pnl_pct >= self.BREAKEVEN_PNL_PCT:
                        stop_loss_price = entry_price
                        self.state[symbol]['stop_loss_price'] = entry_price
                        self._upsert_position_to_db(symbol, state['entry_price'], state['high'], state['low'], state['side'], state['amount'], state['entry_regime'], entry_price)
                        print(f"💰 [{symbol}] 본절 전환! StopLoss -> {entry_price}")

                    should_close = False
                    exit_reason = "Unknown"

                    # ATR 기반 동적 손절
                    if pos_side == 'long' and curr_price < stop_loss_price:
                        should_close = True
                        exit_reason = "StopLoss" if stop_loss_price != entry_price else "Breakeven"
                    elif pos_side == 'short' and curr_price > stop_loss_price:
                        should_close = True
                        exit_reason = "StopLoss" if stop_loss_price != entry_price else "Breakeven"

                    # 트레일링 스탑
                    if not should_close:
                        if pos_side == 'long':
                            new_high = max(state['high'], curr_price)
                            if new_high > state['high']:
                                self.state[symbol]['high'] = new_high
                                self._upsert_position_to_db(symbol, entry_price, new_high, state['low'], pos_side, pos_amt, state['entry_regime'], stop_loss_price)
                            if (new_high - curr_price) / new_high > self.TRAIL_PCT and curr_price > entry_price:
                                should_close = True
                                exit_reason = "TrailingStop"
                        elif pos_side == 'short':
                            new_low = min(state['low'], curr_price)
                            if new_low < state['low']:
                                self.state[symbol]['low'] = new_low
                                self._upsert_position_to_db(symbol, entry_price, state['high'], new_low, pos_side, pos_amt, state['entry_regime'], stop_loss_price)
                            if (curr_price - new_low) / new_low > self.TRAIL_PCT and curr_price < entry_price:
                                should_close = True
                                exit_reason = "TrailingStop"

                    # 국면 전환 청산
                    entry_regime = state.get('entry_regime')
                    if not should_close:
                        if entry_regime == 'bull' and regime == r_map['bear']:
                            should_close = True; exit_reason = "RegimeChange"
                        elif entry_regime == 'bear' and regime == r_map['bull']:
                            should_close = True; exit_reason = "RegimeChange"

                    if should_close:
                        print(f"🔥 [{symbol}] 포지션 청산 ({exit_reason})")
                        await self.execute_order(symbol, 'sell' if pos_side == 'long' else 'buy', pos_amt, reduce_only=True)
                        self._delete_position_from_db(symbol)
                        if symbol in self.state: del self.state[symbol]
                        self.cooldowns[symbol] = self.COOLDOWN_BARS # 쿨다운 설정
                        continue
                
                # --- 2. 진입 로직 (V8) ---
                else:
                    if symbol in self.cooldowns: # 쿨다운 체크
                        print(f"❄️ [{symbol}] 쿨다운 중... ({self.cooldowns[symbol]}시간 남음)")
                        continue

                    await self.set_leverage(symbol, leverage)
                    qty = self.calculate_position_size(usdt_balance, curr_atr)
                    if qty <= 0: continue
                    qty *= leverage
                    
                    side_to_enter, entry_regime_name = None, None

                    if regime == r_map['bull']:
                        if curr_rsi < self.RSI_BUY and curr_price > curr_ema:
                            side_to_enter, entry_regime_name = 'buy', 'bull'
                    elif regime == r_map['bear']:
                        if curr_rsi > self.RSI_SELL and curr_price < curr_ema:
                            side_to_enter, entry_regime_name = 'sell', 'bear'
                    
                    if side_to_enter:
                        exec_price = await self.execute_order(symbol, side_to_enter, qty)
                        if exec_price:
                            stop_loss_price = exec_price - (curr_atr * self.STOP_LOSS_ATR_MULTIPLIER) if side_to_enter == 'buy' else exec_price + (curr_atr * self.STOP_LOSS_ATR_MULTIPLIER)
                            self.state[symbol] = {
                                'entry_price': exec_price, 'high': exec_price, 'low': exec_price,
                                'side': side_to_enter, 'amount': qty, 'entry_regime': entry_regime_name,
                                'stop_loss_price': stop_loss_price
                            }
                            self._upsert_position_to_db(
                                symbol, exec_price, exec_price, exec_price, side_to_enter, qty, entry_regime_name, stop_loss_price
                            )
                            print(f"🚀 [{symbol}] 신규 진입! ({side_to_enter} @ {exec_price}) SL: {stop_loss_price}")

            return "✅ 매매 로직 실행 완료"