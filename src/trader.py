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
        
        if 'entry_regime' not in columns:
            cursor.execute("ALTER TABLE positions ADD COLUMN entry_regime TEXT")
        if 'stop_loss_price' not in columns:
            cursor.execute("ALTER TABLE positions ADD COLUMN stop_loss_price REAL DEFAULT 0.0")
        if 'mode' not in columns:
            cursor.execute("ALTER TABLE positions ADD COLUMN mode TEXT DEFAULT 'UNKNOWN'")

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
        self.state = self._load_positions_from_db()
        self.cooldowns = {}
        
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
        """
        max_slots = CONFIG['MAX_OPEN_POSITIONS']
        
        if current_positions_count >= max_slots:
            return 0.0
            
        per_slot_equity = total_equity / max_slots
        entry_margin = per_slot_equity * 0.95
        
        if entry_margin < 10:
            return 0.0
            
        return entry_margin

    async def fetch_data_and_features(self, symbol):
        try:
            timeframe = CONFIG['TIMEFRAME']
            candles = await self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=CONFIG['CANDLE_LIMIT'])
            if not candles: return None

            last_candle_ts = pd.to_datetime(candles[-1][0], unit='ms', utc=True)
            now_utc = datetime.now(last_candle_ts.tz)
            if now_utc < last_candle_ts + pd.to_timedelta(timeframe):
                candles = candles[:-1]
            if not candles: return None

            df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df.set_index('timestamp', inplace=True)

            # --- Feature Engineering ---
            df['Log_Returns'] = np.log(df['close'] / df['close'].shift(1))
            df['Range_Vol'] = (df['high'] - df['low']) / df['close']
            df.ta.rsi(length=14, append=True)
            df.ta.ema(length=self.EMA_PERIOD, append=True)
            df.ta.atr(length=14, append=True)
            df.ta.obv(append=True)

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
            
            limit_order = await self._create_limit_order_with_retry(symbol, side, amount, limit_price, params)
            order_id = limit_order['id']
            
            start_time = asyncio.get_event_loop().time()
            while asyncio.get_event_loop().time() - start_time < timeout:
                order_status = await self.exchange.fetch_order(order_id, symbol)
                if order_status['status'] == 'closed':
                    return order_status['average']
                await asyncio.sleep(1)

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

        # ---------------------------------------------------------
        # [수정] 1. 모의매매 (PAPER) - 수수료 + 슬리피지 현실 보정
        # ---------------------------------------------------------
        if self.mode == 'PAPER':
            slippage = CONFIG['SLIPPAGE_PCT']
            commission = CONFIG['COMMISSION']
            
            # 총 페널티 (현실 반영)
            total_penalty = slippage + commission
            
            if side == 'buy':
                exec_price = current_price * (1 + total_penalty)
            else:
                exec_price = current_price * (1 - total_penalty)
            
            log_msg = f"🧪 [모의/현실보정] {symbol} {side.upper()} {msg_type} (시장가: {current_price}, 체결가: {exec_price:.4f})"
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
        # [NEW] 모의매매면 가짜 돈 100불 리턴 (실제 계획 중인 시드)
        if self.mode == 'PAPER':
            return 100.0, 100.0

        try:
            balance = await self.exchange.fetch_balance()
            return balance['free']['USDT'], balance['total']['USDT']
        except: return 0.0, 0.0

    async def get_positions(self):
        if self.mode != 'REAL':
            positions = []
            for sym, data in self.state.items():
                # 단순 PnL 계산 (현재가 조회가 없으므로 정확한 PnL은 아님, 단순히 진입 정보 표시용)
                positions.append({
                    'symbol': sym,
                    'side': data['side'],
                    'amount': data['amount'],
                    'entryPrice': data['entry_price'],
                    'unrealizedPnl': 0.0,
                    'leverage': CONFIG['LEVERAGE']
                })
            return positions

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
            try: await self.exchange.cancel_all_orders(symbol)
            except: pass
            
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
            
            for s in list(self.cooldowns.keys()):
                self.cooldowns[s] -= 1
                if self.cooldowns[s] <= 0: del self.cooldowns[s]

            # 자산 조회 (모의매매 가상 자금 주입)
            try:
                if self.mode == 'PAPER':
                    total_equity = 100.0 # 100달러 시드 가정
                else:
                    balance_data = await self.exchange.fetch_balance()
                    total_equity = balance_data['total']['USDT']
            except Exception as e:
                return f"❌ 잔고 조회 실패: {e}"

            current_pos_count = len(self.state)
            
            for symbol in CONFIG['SYMBOLS']:
                df = await self.fetch_data_and_features(symbol)
                if df is None: continue
                
                curr_price = df['close'].iloc[-1]
                curr_rsi = df['RSI_14'].iloc[-1]
                curr_ema = df[f'EMA_{self.EMA_PERIOD}'].iloc[-1]
                curr_atr = df['ATRr_14'].iloc[-1]

                regime, r_map = self.get_hmm_regime(df, symbol)
                if regime is None: continue
                
                has_position = symbol in self.state
                
                # [A] 청산 및 관리 로직
                if has_position:
                    state = self.state[symbol]
                    entry_price = state['entry_price']
                    stop_loss_price = state.get('stop_loss_price', 0)
                    highest_price = state.get('high', entry_price)
                    lowest_price = state.get('low', entry_price)
                    entry_regime = state.get('entry_regime', 'unknown')
                    pos_side = state['side']
                    pos_amt = state['amount']
                    
                    if stop_loss_price == 0:
                        dist = curr_atr * CONFIG['STOP_LOSS_ATR']
                        stop_loss_price = entry_price - dist if pos_side == 'buy' else entry_price + dist
                    
                    should_close = False
                    exit_reason = ""

                    # A-1. 정적 손절
                    if pos_side == 'buy' and curr_price < stop_loss_price:
                        should_close = True; exit_reason = "StopLoss"
                    elif pos_side == 'sell' and curr_price > stop_loss_price:
                        should_close = True; exit_reason = "StopLoss"

                    # A-2. 동적 트레일링 스탑
                    if not should_close:
                        trail_trigger = curr_atr * CONFIG['TRAIL_TRIGGER_ATR'] 
                        trail_dist = curr_atr * CONFIG['TRAIL_DIST_ATR']
                        
                        if pos_side == 'buy':
                            if curr_price > highest_price:
                                highest_price = curr_price
                                self._upsert_position_to_db(symbol, entry_price, highest_price, lowest_price, pos_side, pos_amt, entry_regime, stop_loss_price)
                            
                            if highest_price >= entry_price + trail_trigger:
                                new_stop = highest_price - trail_dist
                                if new_stop > stop_loss_price:
                                    stop_loss_price = new_stop
                                    self._upsert_position_to_db(symbol, entry_price, highest_price, lowest_price, pos_side, pos_amt, entry_regime, stop_loss_price)
                                    print(f"📈 [{symbol}] Trailing Stop 상향 -> {stop_loss_price:.2f}")
                        
                        elif pos_side == 'sell':
                            if curr_price < lowest_price:
                                lowest_price = curr_price
                                self._upsert_position_to_db(symbol, entry_price, highest_price, lowest_price, pos_side, pos_amt, entry_regime, stop_loss_price)
                            
                            if lowest_price <= entry_price - trail_trigger:
                                new_stop = lowest_price + trail_dist
                                if new_stop < stop_loss_price:
                                    stop_loss_price = new_stop
                                    self._upsert_position_to_db(symbol, entry_price, highest_price, lowest_price, pos_side, pos_amt, entry_regime, stop_loss_price)
                                    print(f"📉 [{symbol}] Trailing Stop 하향 -> {stop_loss_price:.2f}")

                    # A-3. 국면 전환 청산
                    if not should_close:
                        if (entry_regime == 'bull' and regime == r_map['bear']) or \
                           (entry_regime == 'bear' and regime == r_map['bull']):
                            should_close = True; exit_reason = f"RegimeChange ({entry_regime}->New)"

                    if should_close:
                        print(f"🔥 [{symbol}] 포지션 청산 ({exit_reason})")
                        close_side = 'sell' if pos_side == 'buy' else 'buy'
                        await self.execute_order(symbol, close_side, pos_amt, reduce_only=True)
                        
                        self._delete_position_from_db(symbol)
                        del self.state[symbol]
                        self.cooldowns[symbol] = self.COOLDOWN_BARS
                        current_pos_count -= 1
                        continue

                # [B] 진입 로직
                else:
                    if current_pos_count >= CONFIG['MAX_OPEN_POSITIONS']: continue
                    if symbol in self.cooldowns: continue 
                    
                    entry_signal = None
                    target_regime = None

                    if regime == r_map['bull']:
                        if curr_price > curr_ema and \
                           CONFIG['RSI_BUY_LOWER'] < curr_rsi < CONFIG['RSI_BUY_UPPER']:
                            entry_signal = 'buy'; target_regime = 'bull'
                    
                    elif regime == r_map['bear']:
                        if curr_price < curr_ema and \
                           CONFIG['RSI_SELL_LOWER'] < curr_rsi < CONFIG['RSI_SELL_UPPER']:
                            entry_signal = 'sell'; target_regime = 'bear'
                    
                    if entry_signal:
                        leverage = CONFIG['LEVERAGE']
                        await self.set_leverage(symbol, leverage)
                        
                        entry_margin = self.calculate_position_size_v2(total_equity, current_pos_count, leverage)
                        if entry_margin == 0: continue
                        
                        qty_contract = (entry_margin * leverage) / curr_price
                        
                        exec_price = await self.execute_order(symbol, entry_signal, qty_contract)
                        
                        if exec_price:
                            dist = curr_atr * CONFIG['STOP_LOSS_ATR']
                            init_sl = exec_price - dist if entry_signal == 'buy' else exec_price + dist
                            
                            self._upsert_position_to_db(
                                symbol, exec_price, exec_price, exec_price, 
                                entry_signal, qty_contract, target_regime, init_sl
                            )
                            self.state[symbol] = {
                                'entry_price': exec_price, 'high': exec_price, 'low': exec_price,
                                'side': entry_signal, 'amount': qty_contract, 
                                'entry_regime': target_regime, 'stop_loss_price': init_sl
                            }
                            current_pos_count += 1

            return "✅ 매매 로직 실행 완료"