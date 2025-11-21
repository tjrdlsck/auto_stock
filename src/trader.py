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
        self._init_db() 
        
        # 매매 모드 설정 (OFF / REAL / PAPER)
        self.mode = self._get_mode_from_db()

        # 전략 내부 상수
        self.EMA_PERIOD = 20
        self.COOLDOWN_BARS = 2  # 진입/청산 후 대기 봉 개수
        
        # 동시성 제어
        self.trade_lock = asyncio.Lock() 
        self.cooldowns = {} 
        
        # 메모리 상태 로드 (현재 모드에 맞는 데이터만)
        self.state = self._load_positions_from_db()

    # -----------------------------------------------------------
    # [DB Section] SQLite Helper Methods (Data Isolation Applied)
    # -----------------------------------------------------------
    def _init_db(self):
        """DB 초기화: 모드(REAL/PAPER) 격리를 위한 스키마 적용 및 마이그레이션"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 1. positions 테이블 생성 (mode 컬럼 포함)
        # (symbol, mode)가 유니크해야 하므로 PK는 아니지만 조회 시 항상 두 조건을 같이 사용함
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS positions (
                symbol TEXT,
                mode TEXT,
                entry_price REAL,
                highest_price REAL,
                lowest_price REAL,
                side TEXT,
                amount REAL,
                entry_regime TEXT,
                stop_loss_price REAL DEFAULT 0.0
            )
        ''')
        
        # 2. system_settings 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_settings (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        ''')
        
        # --- [마이그레이션] 컬럼 추가 ---
        cursor.execute("PRAGMA table_info(positions)")
        columns = [info[1] for info in cursor.fetchall()]
        
        # 필수 컬럼이 없으면 추가
        if 'entry_regime' not in columns:
            cursor.execute("ALTER TABLE positions ADD COLUMN entry_regime TEXT")
        if 'stop_loss_price' not in columns:
            cursor.execute("ALTER TABLE positions ADD COLUMN stop_loss_price REAL DEFAULT 0.0")
        if 'mode' not in columns:
            # 기존 데이터는 출처를 모르므로 'UNKNOWN' 처리하여 로직에서 배제
            cursor.execute("ALTER TABLE positions ADD COLUMN mode TEXT DEFAULT 'UNKNOWN'")

        # 기본 설정값 주입
        cursor.execute("INSERT OR IGNORE INTO system_settings (key, value) VALUES ('mode', 'OFF')")
        
        conn.commit()
        conn.close()

    def _get_mode_from_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT value FROM system_settings WHERE key='mode'")
        result = cursor.fetchone()
        conn.close()
        return result[0] if result else 'OFF'

    def _load_positions_from_db(self):
        """현재 모드(REAL/PAPER)에 해당하는 포지션만 로드"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 격리 조건: WHERE mode = ?
        cursor.execute("""
            SELECT symbol, entry_price, highest_price, lowest_price, side, amount, entry_regime, stop_loss_price 
            FROM positions 
            WHERE mode = ?
        """, (self.mode,))
        
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
        """현재 모드(REAL/PAPER)로 포지션 저장"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 기존 레코드가 있으면 삭제 후 삽입 (SQLite의 INSERT OR REPLACE는 PK 기준이라 복잡하므로 DELETE-INSERT 사용)
        cursor.execute("DELETE FROM positions WHERE symbol = ? AND mode = ?", (symbol, self.mode))
        
        cursor.execute('''
            INSERT INTO positions (symbol, mode, entry_price, highest_price, lowest_price, side, amount, entry_regime, stop_loss_price)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (symbol, self.mode, entry_price, highest_price, lowest_price, side, amount, entry_regime, stop_loss_price))
        
        conn.commit()
        conn.close()

    def _delete_position_from_db(self, symbol):
        """현재 모드(REAL/PAPER)의 포지션만 삭제"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM positions WHERE symbol = ? AND mode = ?", (symbol, self.mode))
        conn.commit()
        conn.close()

    def set_mode(self, new_mode):
        new_mode = new_mode.upper()
        if new_mode not in ['OFF', 'REAL', 'PAPER']:
            return "❌ 잘못된 모드입니다."
        
        self.mode = new_mode
        
        # 모드 변경 시, 메모리 상태도 즉시 교체 (데이터 격리 핵심)
        self.state = self._load_positions_from_db()
        self.cooldowns = {} # 쿨다운 초기화
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("UPDATE system_settings SET value = ? WHERE key = 'mode'", (new_mode,))
        conn.commit()
        conn.close()
        
        return f"✅ 시스템 모드가 **{new_mode}**로 변경되었습니다. (데이터 격리 적용됨)"

    # -----------------------------------------------------------
    # [Logic Section] Trading Helper Methods
    # -----------------------------------------------------------

    async def set_leverage(self, symbol, leverage):
        try:
            await self.exchange.set_leverage(leverage, symbol)
        except Exception:
            pass 

    def calculate_position_size_v2(self, total_equity, current_positions_count, leverage):
        """
        [자금 관리 V2] 포트폴리오 분산 투자
        :param total_equity: 총 자산 (지갑 잔고 + 미실현 손익)
        :param current_positions_count: 현재 보유 중인 포지션 개수
        """
        max_slots = CONFIG['MAX_OPEN_POSITIONS']
        
        # 1. 슬롯 초과 시 진입 불가
        if current_positions_count >= max_slots:
            return 0.0
            
        # 2. 1개 슬롯당 할당 금액 계산 (총 자산 / 최대 슬롯)
        # 예: 자산 1000불, 3슬롯 -> 슬롯당 333불 할당
        per_slot_equity = total_equity / max_slots
        
        # 3. 안전 마진 5% 확보 (수수료 등 대비)
        entry_margin = per_slot_equity * 0.95
        
        # 최소 주문 금액 필터 (10 USDT 미만 진입 불가)
        if entry_margin < 10:
            return 0.0
            
        return entry_margin

    async def fetch_data_and_features(self, symbol):
        try:
            timeframe = CONFIG['TIMEFRAME']
            # 지표 계산을 위해 충분한 데이터 가져오기
            candles = await self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=CONFIG['CANDLE_LIMIT'])
            if not candles: return None

            # 마지막 미완성 캔들 제거 로직
            last_candle_ts = pd.to_datetime(candles[-1][0], unit='ms', utc=True)
            now_utc = datetime.now(last_candle_ts.tz)
            if now_utc < last_candle_ts + pd.to_timedelta(timeframe):
                candles = candles[:-1]
            if not candles: return None

            df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df.set_index('timestamp', inplace=True)

            # --- Feature Engineering (V13 Sync) ---
            df['Log_Returns'] = np.log(df['close'] / df['close'].shift(1))
            df['Range_Vol'] = (df['high'] - df['low']) / df['close']
            df.ta.rsi(length=14, append=True)
            df.ta.ema(length=self.EMA_PERIOD, append=True)
            df.ta.atr(length=14, append=True) # ATR 14 필수
            df.ta.obv(append=True)

            # Scaling for HMM
            window = 30
            features_to_scale = ['Log_Returns', 'Range_Vol', 'RSI_14', 'OBV']
            for col in features_to_scale:
                rolling_mean = df[col].rolling(window=window).mean()
                rolling_std = df[col].rolling(window=window).std()
                df[f'{col}_Scaled'] = (df[col] - rolling_mean) / (rolling_std + 1e-8)
            
            return df.dropna()
        except Exception as e:
            print(f"❌ 데이터 처리 에러 ({symbol}): {e}")
            return None

    def get_hmm_regime(self, df, symbol):
        try:
            clean_symbol = symbol.replace('/', '')
            model_path = os.path.join(MODELS_DIR, f"hmm_{clean_symbol}.pkl")
            if not os.path.exists(model_path): return None, None

            model = joblib.load(model_path)
            X = df[['Log_Returns_Scaled', 'Range_Vol_Scaled', 'RSI_14_Scaled', 'OBV_Scaled']].values
            hidden_states = model.predict(X)
            
            # Regime Definition
            stats = pd.DataFrame(X, columns=['Ret', 'Vol', 'RSI', 'OBV'])
            stats['Regime'] = hidden_states
            regime_means = stats.groupby('Regime')['Ret'].mean()
            
            bear_id = regime_means.idxmin()
            bull_id = regime_means.idxmax()
            sideways_id = list(set(range(model.n_components)) - {bull_id, bear_id})[0] if model.n_components > 2 else None
            
            return hidden_states[-1], {'bull': bull_id, 'bear': bear_id, 'sideways': sideways_id}
        except:
            return None, None

    # -----------------------------------------------------------
    # [Execution Section] Smart Order & Retry
    # -----------------------------------------------------------
    @retry(stop=stop_after_attempt(3), wait=wait_fixed(1), retry=retry_if_exception_type((ccxt.NetworkError, ccxt.RequestTimeout, ccxt.ExchangeError)))
    async def _create_market_order_with_retry(self, symbol, side, amount, params):
        return await self.exchange.create_market_order(symbol, side, amount, params)

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(1), retry=retry_if_exception_type((ccxt.NetworkError, ccxt.RequestTimeout, ccxt.ExchangeError)))
    async def _create_limit_order_with_retry(self, symbol, side, amount, price, params):
        return await self.exchange.create_limit_order(symbol, side, amount, price, params)

    async def _smart_execute_order(self, symbol, side, amount, reduce_only=False):
        params = {'reduceOnly': True} if reduce_only else {}
        timeout = CONFIG['LIMIT_ORDER_TIMEOUT_SEC']
        force_market = CONFIG['FORCE_MARKET_ORDER_ON_TIMEOUT']

        try:
            orderbook = await self.exchange.fetch_order_book(symbol)
            limit_price = orderbook['asks'][0][0] if side == 'buy' else orderbook['bids'][0][0]
            
            # 1. 지정가 주문 시도
            limit_order = await self._create_limit_order_with_retry(symbol, side, amount, limit_price, params)
            order_id = limit_order['id']
            
            # 2. 체결 대기
            start_time = asyncio.get_event_loop().time()
            while asyncio.get_event_loop().time() - start_time < timeout:
                order_status = await self.exchange.fetch_order(order_id, symbol)
                if order_status['status'] == 'closed':
                    return order_status['average']
                await asyncio.sleep(1)

            # 3. 타임아웃: 취소 후 시장가 전환
            await self.exchange.cancel_order(order_id, symbol)
            if force_market:
                market_order = await self._create_market_order_with_retry(symbol, side, amount, params)
                return market_order['average']
            return None
        except Exception as e:
            print(f"❌ 주문 실행 에러 ({symbol}): {e}")
            return None

    async def execute_order(self, symbol, side, amount, reduce_only=False):
        if self.mode == 'OFF': return None

        try:
            ticker = await self.exchange.fetch_ticker(symbol)
            current_price = ticker['last']
        except: return None

        msg_type = "청산" if reduce_only else "진입"

        # 1. 모의매매 (PAPER)
        if self.mode == 'PAPER':
            slippage = CONFIG['SLIPPAGE_PCT']
            # 슬리피지 적용된 가상 체결가
            exec_price = current_price * (1 + slippage) if side == 'buy' else current_price * (1 - slippage)
            
            log_msg = f"🧪 [모의] {symbol} {side.upper()} {msg_type} (가격: {exec_price:.4f}, 수량: {amount:.4f})"
            print(log_msg)
            if self.messenger: self.messenger(symbol, side, exec_price, amount, f"[모의] {msg_type}")
            return exec_price

        # 2. 실매매 (REAL)
        if self.mode == 'REAL':
            exec_price = await self._smart_execute_order(symbol, side, amount, reduce_only)
            if exec_price:
                print(f"🚀 [실매매] {symbol} {side} 체결 (가격: {exec_price})")
                if self.messenger: self.messenger(symbol, side, exec_price, amount, msg_type)
                return exec_price
            else:
                if self.messenger: self.messenger(symbol, side, 0, amount, "[긴급] 주문 실패", mention_everyone=True)
                return None

    # -----------------------------------------------------------
    # [Info Section] Status Check
    # -----------------------------------------------------------
    async def get_balance(self):
        # [NEW] 모의매매면 가짜 돈 100불 리턴
        if self.mode == 'PAPER':
            return 100.0, 100.0

        try:
            balance = await self.exchange.fetch_balance()
            return balance['free']['USDT'], balance['total']['USDT']
        except: return 0.0, 0.0

    async def get_positions(self):
        """실매매(REAL) 모드일 때만 거래소 API 포지션 반환"""
        if self.mode != 'REAL':
            # PAPER 모드일 땐 DB에 있는 내용을 포지션 객체처럼 변환해서 반환
            positions = []
            for sym, data in self.state.items():
                pnl = (data['high'] - data['entry_price']) * data['amount'] if data['side'] == 'buy' else (data['entry_price'] - data['low']) * data['amount']
                # 단순화된 PNL 계산
                positions.append({
                    'symbol': sym,
                    'side': data['side'],
                    'amount': data['amount'],
                    'entryPrice': data['entry_price'],
                    'unrealizedPnl': 0.0, # 모의매매 실시간 PnL 계산은 복잡하므로 0 처리하거나 추후 구현
                    'leverage': CONFIG['LEVERAGE']
                })
            return positions

        # REAL 모드
        active_positions = []
        try:
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
        except: return []

    async def safe_force_close(self, symbol):
        async with self.trade_lock:
            # 1. 미체결 취소
            try: await self.exchange.cancel_all_orders(symbol)
            except: pass
            
            # 2. 포지션 확인 및 청산
            # DB 상태 우선 확인
            if symbol in self.state:
                data = self.state[symbol]
                side = 'sell' if data['side'] == 'buy' else 'buy'
                amount = data['amount']
                
                res = await self.execute_order(symbol, side, amount, reduce_only=True)
                if res:
                    self._delete_position_from_db(symbol)
                    del self.state[symbol]
                    return f"✅ {symbol} 강제 청산 완료 (Mode: {self.mode})"
            
            return f"⚠️ {symbol} 청산할 포지션 없음 (Mode: {self.mode})"

    # -----------------------------------------------------------
    # [Main Logic] V13 Strategy + Portfolio Mgmt + DB Isolation
    # -----------------------------------------------------------
    async def run_logic(self):
        async with self.trade_lock:
            if self.mode == 'OFF':
                return "⏸️ 봇 정지 상태"
            
            # 쿨다운 감소
            for s in list(self.cooldowns.keys()):
                self.cooldowns[s] -= 1
                if self.cooldowns[s] <= 0: del self.cooldowns[s]

            try:
                if self.mode == 'PAPER':
                    # 모의매매면 100 달러가 있다고 가정 (고정)
                    total_equity = 100.0
                    # 디스코드 표시용 변수는 별도로 안 쓰이지만 로그 확인용
                    print(f"🧪 [모의] 가상 자산 $10,000 적용됨")
                else:
                    # 실매매면 실제 바이낸스 잔고 조회
                    balance_data = await self.exchange.fetch_balance()
                    total_equity = balance_data['total']['USDT']
            except Exception as e:
                return f"❌ 잔고 조회 실패: {e}"

            # 1. 현재 보유 포지션 동기화 (메모리 vs 실제)
            # PAPER 모드: DB(self.state)가 곧 진실
            # REAL 모드: API 조회 결과와 DB를 비교해야 하지만, 여기선 DB를 기준으로 하되 API와 대조하는 로직은 생략하고 DB를 믿고 감.
            # (실매매 시 봇 외부에서 포지션 건드리면 안됨)
            
            current_pos_count = len(self.state)
            
            # ---------------- Loop Start ----------------
            for symbol in CONFIG['SYMBOLS']:
                # 데이터 준비
                df = await self.fetch_data_and_features(symbol)
                if df is None: continue
                
                # 현재 지표 값
                curr_price = df['close'].iloc[-1]
                curr_rsi = df['RSI_14'].iloc[-1]
                curr_ema = df[f'EMA_{self.EMA_PERIOD}'].iloc[-1]
                curr_atr = df['ATRr_14'].iloc[-1]

                # 국면 식별
                regime, r_map = self.get_hmm_regime(df, symbol)
                if regime is None: continue
                
                # 포지션 보유 여부
                has_position = symbol in self.state
                
                # ============================================
                # [A] 청산 및 관리 로직 (보유 중일 때)
                # ============================================
                if has_position:
                    state = self.state[symbol]
                    entry_price = state['entry_price']
                    stop_loss_price = state.get('stop_loss_price', 0)
                    highest_price = state.get('high', entry_price)
                    lowest_price = state.get('low', entry_price)
                    entry_regime = state.get('entry_regime', 'unknown')
                    pos_side = state['side']
                    pos_amt = state['amount']
                    
                    # 안전장치: SL이 0이면 초기화
                    if stop_loss_price == 0:
                        dist = curr_atr * CONFIG['STOP_LOSS_ATR']
                        stop_loss_price = entry_price - dist if pos_side == 'buy' else entry_price + dist
                    
                    should_close = False
                    exit_reason = ""

                    # A-1. 정적 손절 (Static SL)
                    if pos_side == 'buy' and curr_price < stop_loss_price:
                        should_close = True; exit_reason = "StopLoss"
                    elif pos_side == 'sell' and curr_price > stop_loss_price:
                        should_close = True; exit_reason = "StopLoss"

                    # A-2. 동적 트레일링 스탑 (V13 Dynamic ATR Trailing)
                    if not should_close:
                        trail_trigger = curr_atr * CONFIG['TRAIL_TRIGGER_ATR'] 
                        trail_dist = curr_atr * CONFIG['TRAIL_DIST_ATR']
                        
                        if pos_side == 'buy':
                            # 고점 갱신
                            if curr_price > highest_price:
                                highest_price = curr_price
                                self._upsert_position_to_db(symbol, entry_price, highest_price, lowest_price, pos_side, pos_amt, entry_regime, stop_loss_price)
                            
                            # 트레일링 발동: (최고가 - 진입가) > Trigger
                            if highest_price >= entry_price + trail_trigger:
                                new_stop = highest_price - trail_dist
                                # 스탑 상향 조정만 허용
                                if new_stop > stop_loss_price:
                                    stop_loss_price = new_stop
                                    self._upsert_position_to_db(symbol, entry_price, highest_price, lowest_price, pos_side, pos_amt, entry_regime, stop_loss_price)
                                    print(f"📈 [{symbol}] Trailing Stop 상향 -> {stop_loss_price:.2f}")
                        
                        elif pos_side == 'sell':
                            # 저점 갱신
                            if curr_price < lowest_price:
                                lowest_price = curr_price
                                self._upsert_position_to_db(symbol, entry_price, highest_price, lowest_price, pos_side, pos_amt, entry_regime, stop_loss_price)
                            
                            # 트레일링 발동
                            if lowest_price <= entry_price - trail_trigger:
                                new_stop = lowest_price + trail_dist
                                # 스탑 하향 조정만 허용
                                if new_stop < stop_loss_price:
                                    stop_loss_price = new_stop
                                    self._upsert_position_to_db(symbol, entry_price, highest_price, lowest_price, pos_side, pos_amt, entry_regime, stop_loss_price)
                                    print(f"📉 [{symbol}] Trailing Stop 하향 -> {stop_loss_price:.2f}")

                    # A-3. 국면 전환 청산 (Regime Change)
                    if not should_close:
                        if (entry_regime == 'bull' and regime == r_map['bear']) or \
                           (entry_regime == 'bear' and regime == r_map['bull']):
                            should_close = True; exit_reason = f"RegimeChange ({entry_regime}->New)"

                    # 실행
                    if should_close:
                        print(f"🔥 [{symbol}] 포지션 청산 ({exit_reason})")
                        # 청산 주문
                        close_side = 'sell' if pos_side == 'buy' else 'buy'
                        await self.execute_order(symbol, close_side, pos_amt, reduce_only=True)
                        
                        # DB 및 메모리 삭제
                        self._delete_position_from_db(symbol)
                        del self.state[symbol]
                        
                        # 쿨다운 및 슬롯 반환
                        self.cooldowns[symbol] = self.COOLDOWN_BARS
                        current_pos_count -= 1
                        continue

                # ============================================
                # [B] 진입 로직 (미보유 일 때)
                # ============================================
                else:
                    # B-1. 자금 관리 & 필터
                    if current_pos_count >= CONFIG['MAX_OPEN_POSITIONS']: continue # 슬롯 가득 참
                    if symbol in self.cooldowns: continue 
                    
                    # B-2. 진입 시그널 (V13 Strategy)
                    entry_signal = None
                    target_regime = None

                    # Bull: Price > EMA & RSI [30, 65]
                    if regime == r_map['bull']:
                        if curr_price > curr_ema and \
                           CONFIG['RSI_BUY_LOWER'] < curr_rsi < CONFIG['RSI_BUY_UPPER']:
                            entry_signal = 'buy'; target_regime = 'bull'
                    
                    # Bear: Price < EMA & RSI [35, 70]
                    elif regime == r_map['bear']:
                        if curr_price < curr_ema and \
                           CONFIG['RSI_SELL_LOWER'] < curr_rsi < CONFIG['RSI_SELL_UPPER']:
                            entry_signal = 'sell'; target_regime = 'bear'
                    
                    # B-3. 진입 실행
                    if entry_signal:
                        leverage = CONFIG['LEVERAGE']
                        await self.set_leverage(symbol, leverage)
                        
                        # 포트폴리오 분산 금액 계산
                        entry_margin = self.calculate_position_size_v2(total_equity, current_pos_count, leverage)
                        if entry_margin == 0: continue
                        
                        qty_contract = (entry_margin * leverage) / curr_price
                        
                        # 주문 실행
                        exec_price = await self.execute_order(symbol, entry_signal, qty_contract)
                        
                        if exec_price:
                            # 초기 손절가 계산
                            dist = curr_atr * CONFIG['STOP_LOSS_ATR']
                            init_sl = exec_price - dist if entry_signal == 'buy' else exec_price + dist
                            
                            # DB 저장 (mode 포함)
                            self._upsert_position_to_db(
                                symbol, exec_price, exec_price, exec_price, 
                                entry_signal, qty_contract, target_regime, init_sl
                            )
                            # 메모리 업데이트
                            self.state[symbol] = {
                                'entry_price': exec_price, 'high': exec_price, 'low': exec_price,
                                'side': entry_signal, 'amount': qty_contract, 
                                'entry_regime': target_regime, 'stop_loss_price': init_sl
                            }
                            current_pos_count += 1 # 루프 내 카운트 증가

            return "✅ 매매 로직 실행 완료"