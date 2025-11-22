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
from config import MODELS_DIR, DATA_DIR

# .env 경로 명시적 로드
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
ENV_PATH = os.path.join(ROOT_DIR, '.env')
load_dotenv(dotenv_path=ENV_PATH)

IS_TESTNET = os.getenv("TRADING_MODE_TESTNET") == 'True'

class BinanceTrader:
    def __init__(self, config_manager, notification_hub):
        self.config_manager = config_manager
        self.notification = notification_hub
        
        # ---------------------------------------------------
        # [키 로드 및 보정 로직]
        # ---------------------------------------------------
        raw_api_key = os.getenv("BINANCE_API_KEY", "").strip()
        raw_secret_key = os.getenv("BINANCE_SECRET_KEY", "").strip()

        if len(raw_secret_key) > 64:
            raw_secret_key = raw_secret_key[:64]

        print("="*40)
        print(f"🆔 API KEY : {raw_api_key[:4]}**** (길이: {len(raw_api_key)})")
        print(f"🔑 SECRET  : {raw_secret_key[:4]}**** (길이: {len(raw_secret_key)})")
        print(f"🧪 테스트넷 : {IS_TESTNET}")
        print("="*40)
        
        if len(raw_api_key) < 10 or len(raw_secret_key) < 10:
            print("❌ [CRITICAL] API 키가 로드되지 않았습니다! .env 경로를 확인하세요.")

        # 거래소 연결
        self.exchange = ccxt.binance({
            'apiKey': raw_api_key,
            'secret': raw_secret_key,
            'options': {
                'defaultType': 'future',
                'adjustForTimeDifference': True 
            },
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
        self.COOLDOWN_BARS = 2
        
        # 동시성 제어
        self.trade_lock = asyncio.Lock() 
        self.cooldowns = {} 
        
        # 메모리 상태 로드
        self.state = self._load_positions_from_db()

    def get_conf(self, key, default=None):
        return self.config_manager.get(key, default)

    # -----------------------------------------------------------
    # [DB Section] SQLite Helper Methods
    # -----------------------------------------------------------
    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 1. 활성 포지션 테이블
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
        
        # 2. 시스템 설정 테이블 (모드, 모의투자 잔고 등 저장)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_settings (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        ''')

        # 3. 거래 이력 테이블 (History) - 수수료, PnL 포함
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trade_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                symbol TEXT,
                mode TEXT,
                side TEXT,
                entry_price REAL,
                exit_price REAL,
                amount REAL,
                fee REAL,
                pnl REAL
            )
        ''')
        
        # 마이그레이션 (필드 추가 확인)
        cursor.execute("PRAGMA table_info(positions)")
        columns = [info[1] for info in cursor.fetchall()]
        if 'entry_regime' not in columns:
            cursor.execute("ALTER TABLE positions ADD COLUMN entry_regime TEXT")
        if 'stop_loss_price' not in columns:
            cursor.execute("ALTER TABLE positions ADD COLUMN stop_loss_price REAL DEFAULT 0.0")
        if 'mode' not in columns:
            cursor.execute("ALTER TABLE positions ADD COLUMN mode TEXT DEFAULT 'UNKNOWN'")

        # 기본 모드 설정
        cursor.execute("INSERT OR IGNORE INTO system_settings (key, value) VALUES ('mode', 'OFF')")
        conn.commit()
        conn.close()

    def _get_setting(self, key):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT value FROM system_settings WHERE key=?", (key,))
        res = cursor.fetchone()
        conn.close()
        return res[0] if res else None

    def _set_setting(self, key, value):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("INSERT OR REPLACE INTO system_settings (key, value) VALUES (?, ?)", (key, str(value)))
        conn.commit()
        conn.close()

    def _get_mode_from_db(self):
        mode = self._get_setting('mode')
        return mode if mode else 'OFF'

    def _load_positions_from_db(self):
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
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM positions WHERE symbol = ? AND mode = ?", (symbol, self.mode))
        conn.commit()
        conn.close()

    # -----------------------------------------------------------
    # [중요] 거래 이력 저장 및 통계 조회
    # -----------------------------------------------------------
    def _save_trade_history(self, symbol, side, entry_price, exit_price, amount):
        """청산 시 거래 이력 저장 및 모의투자 잔고 업데이트"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        commission_rate = self.get_conf('COMMISSION', 0.0005)
        entry_fee = entry_price * amount * commission_rate
        exit_fee = exit_price * amount * commission_rate
        total_fee = entry_fee + exit_fee
        
        # PnL 계산
        if side == 'buy': # Long 청산 (매도)
            raw_pnl = (exit_price - entry_price) * amount
        else: # Short 청산 (매수)
            raw_pnl = (entry_price - exit_price) * amount
            
        net_pnl = raw_pnl - total_fee
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # 1. 거래 이력 저장
        cursor.execute('''
            INSERT INTO trade_history (timestamp, symbol, mode, side, entry_price, exit_price, amount, fee, pnl)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (timestamp, symbol, self.mode, side, entry_price, exit_price, amount, total_fee, net_pnl))
        
        # 2. 모의투자일 경우 DB 잔고 업데이트 (영구 저장)
        if self.mode == 'PAPER':
            try:
                cursor.execute("SELECT value FROM system_settings WHERE key='paper_balance'")
                res = cursor.fetchone()
                current_bal = float(res[0]) if res else 10000.0
                
                new_bal = current_bal + net_pnl
                cursor.execute("INSERT OR REPLACE INTO system_settings (key, value) VALUES ('paper_balance', ?)", (str(new_bal),))
                print(f"💰 [Paper] 잔고 업데이트: ${current_bal:.2f} -> ${new_bal:.2f} (PnL: {net_pnl:+.2f})")
            except Exception as e:
                print(f"❌ 모의투자 잔고 업데이트 오류: {e}")

        conn.commit()
        conn.close()

        return net_pnl, total_fee

    def get_trade_history(self, target_mode=None):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        query = "SELECT * FROM trade_history"
        params = []
        if target_mode:
            query += " WHERE mode = ?"
            params.append(target_mode)
        query += " ORDER BY id ASC"
        
        cursor.execute(query, tuple(params))
        rows = cursor.fetchall()
        conn.close()
        
        trades = [dict(row) for row in rows]
        total_pnl = 0.0
        win_count = 0
        max_pnl = 0.0
        equity_curve = []
        cumulative_pnl = 0.0
        
        for t in trades:
            pnl = t['pnl']
            cumulative_pnl += pnl
            total_pnl += pnl
            if pnl > 0: win_count += 1
            if pnl > max_pnl: max_pnl = pnl
            equity_curve.append({"time": t['timestamp'], "value": cumulative_pnl})

        total_trades = len(trades)
        win_rate = round((win_count / total_trades * 100), 1) if total_trades > 0 else 0.0
        
        summary = {
            "total_pnl": round(total_pnl, 2),
            "win_rate": win_rate,
            "total_trades": total_trades,
            "best_trade": round(max_pnl, 2)
        }

        return {
            "summary": summary,
            "equity_curve": equity_curve,
            "trades": trades[::-1]
        }

    def get_saved_paper_balance(self):
        val = self._get_setting('paper_balance')
        return float(val) if val else 10000.0

    def reset_paper_balance(self, initial_balance=10000.0):
        self._set_setting('paper_balance', initial_balance)
        return float(initial_balance)

    async def set_mode(self, new_mode):
        new_mode = new_mode.upper()
        if new_mode not in ['OFF', 'REAL', 'PAPER']:
            return "❌ 잘못된 모드입니다."
        
        self.mode = new_mode
        self.state = self._load_positions_from_db()
        self.cooldowns = {}
        
        self._set_setting('mode', new_mode)
        
        msg = f"✅ 시스템 모드가 **{new_mode}**로 변경되었습니다."
        await self.notification.log(msg, level="SYSTEM")
        return msg

    # -----------------------------------------------------------
    # [Logic Section] Trading Helper Methods
    # -----------------------------------------------------------
    async def set_leverage(self, symbol, leverage):
        try:
            await self.exchange.set_leverage(leverage, symbol)
        except Exception:
            pass 

    def calculate_position_size_v2(self, total_equity, current_positions_count, leverage):
        max_slots = self.get_conf('MAX_OPEN_POSITIONS', 3)
        if current_positions_count >= max_slots: return 0.0
        per_slot_equity = total_equity / max_slots
        entry_margin = per_slot_equity * 0.95
        if entry_margin < 10: return 0.0
        return entry_margin

    async def fetch_data_and_features(self, symbol):
        try:
            timeframe = self.get_conf('TIMEFRAME', '1h')
            limit = self.get_conf('CANDLE_LIMIT', 300)
            
            candles = await self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            if not candles: return None

            last_candle_ts = pd.to_datetime(candles[-1][0], unit='ms', utc=True)
            now_utc = datetime.now(last_candle_ts.tz)
            if now_utc < last_candle_ts + pd.to_timedelta(timeframe):
                candles = candles[:-1]
            if not candles: return None

            df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df.set_index('timestamp', inplace=True)

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
    # [Execution Section]
    # -----------------------------------------------------------
    @retry(stop=stop_after_attempt(3), wait=wait_fixed(1), retry=retry_if_exception_type((ccxt.NetworkError, ccxt.RequestTimeout, ccxt.ExchangeError)))
    async def _create_market_order_with_retry(self, symbol, side, amount, params):
        return await self.exchange.create_market_order(symbol, side, amount, params)

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(1), retry=retry_if_exception_type((ccxt.NetworkError, ccxt.RequestTimeout, ccxt.ExchangeError)))
    async def _create_limit_order_with_retry(self, symbol, side, amount, price, params):
        return await self.exchange.create_limit_order(symbol, side, amount, price, params)

    async def _smart_execute_order(self, symbol, side, amount, reduce_only=False):
        params = {'reduceOnly': True} if reduce_only else {}
        timeout = self.get_conf('LIMIT_ORDER_TIMEOUT_SEC', 10)
        force_market = self.get_conf('FORCE_MARKET_ORDER_ON_TIMEOUT', True)

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
            await self.notification.log(f"❌ 주문 실행 에러 ({symbol}): {e}", level="ERROR")
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
            slippage = self.get_conf('SLIPPAGE_PCT', 0.0002)
            if side == 'buy':
                exec_price = current_price * (1 + slippage)
            else:
                exec_price = current_price * (1 - slippage)
            
            log_msg = f"🧪 [모의] {symbol} {side.upper()} {msg_type} ({exec_price:.4f})"
            await self.notification.log(log_msg, level="PAPER")
            return exec_price

        # 2. 실매매 (REAL)
        if self.mode == 'REAL':
            exec_price = await self._smart_execute_order(symbol, side, amount, reduce_only)
            if exec_price:
                log_msg = f"🚀 [실매매] {symbol} {side.upper()} {msg_type} ({exec_price})"
                await self.notification.log(log_msg, level="REAL")
                return exec_price
            else:
                await self.notification.log(f"🚨 [긴급] {symbol} {side} 주문 실패", level="ERROR")
                return None

    # -----------------------------------------------------------
    # [Info Section]
    # -----------------------------------------------------------
    async def get_balance(self):
        if self.mode == 'PAPER':
            try:
                saved_balance = self._get_setting('paper_balance')
                if saved_balance:
                    return float(saved_balance), float(saved_balance)
                else:
                    self._set_setting('paper_balance', 10000.0)
                    return 10000.0, 10000.0
            except:
                return 10000.0, 10000.0

        try:
            balance = await self.exchange.fetch_balance()
            return balance['free']['USDT'], balance['total']['USDT']
        except: return 0.0, 0.0

    async def get_positions(self):
        if self.mode != 'REAL':
            positions = []
            leverage = self.get_conf('LEVERAGE', 1.0)
            for sym, data in self.state.items():
                positions.append({
                    'symbol': sym,
                    'side': data['side'],
                    'amount': data['amount'],
                    'entryPrice': data['entry_price'],
                    'unrealizedPnl': 0.0, 
                    'leverage': leverage,
                    'mode': self.mode
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
                        'leverage': p['leverage'],
                        'mode': 'REAL'
                    })
            return active_positions
        except: return []

    # -----------------------------------------------------------
    # [청산 로직 - 수정됨]
    # -----------------------------------------------------------
    async def safe_force_close(self, symbol):
        """단일 포지션 강제 청산 (안전 로직 포함)"""
        symbol = symbol.strip() # 안전장치: 공백 제거
        
        async with self.trade_lock:
            # 1. 미체결 주문 취소 시도
            try: await self.exchange.cancel_all_orders(symbol)
            except: pass
            
            # 2. 포지션 데이터 확인
            if symbol not in self.state:
                msg = f"⚠️ {symbol} 청산 실패: 관리 중인 포지션이 아닙니다."
                await self.notification.log(msg, level="WARN")
                return msg

            data = self.state[symbol]
            side = 'sell' if data['side'] == 'buy' else 'buy'
            amount = data['amount']
            
            # 3. 주문 실행 (reduce_only=True)
            exec_price = await self.execute_order(symbol, side, amount, reduce_only=True)
            
            # 4. 결과 처리
            if exec_price:
                # 이력 저장 및 잔고 업데이트
                pnl, fee = self._save_trade_history(symbol, data['side'], data['entry_price'], exec_price, amount)
                
                self._delete_position_from_db(symbol)
                del self.state[symbol]
                
                msg = f"✅ {symbol} 청산 완료. PnL: ${pnl:.2f}"
                await self.notification.log(msg, level="INFO")
                return msg
            else:
                msg = f"❌ {symbol} 청산 주문 실패 (API 오류 또는 잔고 부족)"
                await self.notification.log(msg, level="ERROR")
                return msg
    
    async def close_all_positions(self):
        """현재 보유 중인 모든 포지션 일괄 청산 (Stop 시 호출됨)"""
        if not self.state:
            return "ℹ️ 청산할 포지션이 없습니다."

        # 락 없이 진행 (safe_force_close가 내부적으로 락을 사용하므로)
        # state의 키를 리스트로 복사 (순회 중 삭제 방지)
        symbols = list(self.state.keys())
        
        await self.notification.log(f"🚨 시스템 정지로 인한 일괄 청산 시작 ({len(symbols)}개)", level="SYSTEM")
        
        results = []
        for symbol in symbols:
            res = await self.safe_force_close(symbol)
            results.append(res)
            # 너무 빠른 연속 API 호출 방지
            await asyncio.sleep(0.2) 
            
        return "\n".join(results)

    async def run_logic(self):
        async with self.trade_lock:
            if self.mode == 'OFF': return "⏸️ 봇 정지 상태"
            
            for s in list(self.cooldowns.keys()):
                self.cooldowns[s] -= 1
                if self.cooldowns[s] <= 0: del self.cooldowns[s]

            free_equity, total_equity = await self.get_balance()
            current_pos_count = len(self.state)
            target_symbols = self.get_conf('SYMBOLS', [])
            
            for symbol in target_symbols:
                df = await self.fetch_data_and_features(symbol)
                if df is None: continue
                
                curr_price = df['close'].iloc[-1]
                curr_rsi = df['RSI_14'].iloc[-1]
                curr_ema = df[f'EMA_{self.EMA_PERIOD}'].iloc[-1]
                curr_atr = df['ATRr_14'].iloc[-1]

                regime, r_map = self.get_hmm_regime(df, symbol)
                if regime is None: continue
                
                has_position = symbol in self.state
                
                # ---------------------
                # 청산 로직
                # ---------------------
                if has_position:
                    state = self.state[symbol]
                    entry_price = state['entry_price']
                    stop_loss_price = state.get('stop_loss_price', 0)
                    highest_price = state.get('high', entry_price)
                    lowest_price = state.get('low', entry_price)
                    entry_regime = state.get('entry_regime', 'unknown')
                    pos_side = state['side']
                    pos_amt = state['amount']
                    
                    stop_loss_atr = self.get_conf('STOP_LOSS_ATR', 2.0)
                    
                    if stop_loss_price == 0:
                        dist = curr_atr * stop_loss_atr
                        stop_loss_price = entry_price - dist if pos_side == 'buy' else entry_price + dist
                    
                    should_close = False
                    exit_reason = ""

                    # 1. 스탑로스
                    if pos_side == 'buy' and curr_price < stop_loss_price: should_close = True; exit_reason = "StopLoss"
                    elif pos_side == 'sell' and curr_price > stop_loss_price: should_close = True; exit_reason = "StopLoss"

                    # 2. 트레일링 스탑
                    if not should_close:
                        trail_trigger = curr_atr * self.get_conf('TRAIL_TRIGGER_ATR', 2.0)
                        trail_dist = curr_atr * self.get_conf('TRAIL_DIST_ATR', 2.0)
                        
                        if pos_side == 'buy':
                            if curr_price > highest_price:
                                highest_price = curr_price
                                self._upsert_position_to_db(symbol, entry_price, highest_price, lowest_price, pos_side, pos_amt, entry_regime, stop_loss_price)
                            if highest_price >= entry_price + trail_trigger:
                                new_stop = highest_price - trail_dist
                                if new_stop > stop_loss_price:
                                    stop_loss_price = new_stop
                                    self._upsert_position_to_db(symbol, entry_price, highest_price, lowest_price, pos_side, pos_amt, entry_regime, stop_loss_price)
                        elif pos_side == 'sell':
                            if curr_price < lowest_price:
                                lowest_price = curr_price
                                self._upsert_position_to_db(symbol, entry_price, highest_price, lowest_price, pos_side, pos_amt, entry_regime, stop_loss_price)
                            if lowest_price <= entry_price - trail_trigger:
                                new_stop = lowest_price + trail_dist
                                if new_stop < stop_loss_price:
                                    stop_loss_price = new_stop
                                    self._upsert_position_to_db(symbol, entry_price, highest_price, lowest_price, pos_side, pos_amt, entry_regime, stop_loss_price)

                    # 3. 국면 변화
                    if not should_close:
                        if (entry_regime == 'bull' and regime == r_map['bear']) or \
                           (entry_regime == 'bear' and regime == r_map['bull']):
                            should_close = True; exit_reason = "RegimeChange"

                    if should_close:
                        await self.notification.log(f"🔥 [{symbol}] 청산 신호 ({exit_reason})", level="INFO")
                        close_side = 'sell' if pos_side == 'buy' else 'buy'
                        exec_price = await self.execute_order(symbol, close_side, pos_amt, reduce_only=True)
                        
                        if exec_price:
                            pnl, fee = self._save_trade_history(symbol, pos_side, entry_price, exec_price, pos_amt)
                            await self.notification.log(f"💰 실현 손익: ${pnl:.2f}", level="INFO")
                            
                            self._delete_position_from_db(symbol)
                            del self.state[symbol]
                            self.cooldowns[symbol] = self.COOLDOWN_BARS
                            current_pos_count -= 1
                        continue

                # ---------------------
                # 진입 로직
                # ---------------------
                else:
                    max_pos = self.get_conf('MAX_OPEN_POSITIONS', 3)
                    if current_pos_count >= max_pos: continue
                    if symbol in self.cooldowns: continue 
                    
                    entry_signal = None
                    target_regime = None
                    
                    rsi_buy_lower = self.get_conf('RSI_BUY_LOWER', 30)
                    rsi_buy_upper = self.get_conf('RSI_BUY_UPPER', 65)
                    rsi_sell_lower = self.get_conf('RSI_SELL_LOWER', 35)
                    rsi_sell_upper = self.get_conf('RSI_SELL_UPPER', 70)

                    if regime == r_map['bull']:
                        if curr_price > curr_ema and rsi_buy_lower < curr_rsi < rsi_buy_upper:
                            entry_signal = 'buy'; target_regime = 'bull'
                    elif regime == r_map['bear']:
                        if curr_price < curr_ema and rsi_sell_lower < curr_rsi < rsi_sell_upper:
                            entry_signal = 'sell'; target_regime = 'bear'
                    
                    if entry_signal:
                        leverage = self.get_conf('LEVERAGE', 1.0)
                        await self.set_leverage(symbol, leverage)
                        
                        entry_margin = self.calculate_position_size_v2(total_equity, current_pos_count, leverage)
                        if entry_margin == 0: continue
                        
                        qty_contract = (entry_margin * leverage) / curr_price
                        exec_price = await self.execute_order(symbol, entry_signal, qty_contract)
                        
                        if exec_price:
                            stop_loss_atr = self.get_conf('STOP_LOSS_ATR', 2.0)
                            dist = curr_atr * stop_loss_atr
                            init_sl = exec_price - dist if entry_signal == 'buy' else exec_price + dist
                            
                            self._upsert_position_to_db(symbol, exec_price, exec_price, exec_price, entry_signal, qty_contract, target_regime, init_sl)
                            self.state[symbol] = {
                                'entry_price': exec_price, 'high': exec_price, 'low': exec_price,
                                'side': entry_signal, 'amount': qty_contract, 
                                'entry_regime': target_regime, 'stop_loss_price': init_sl
                            }
                            current_pos_count += 1

            return "✅ 매매 로직 실행 완료"

    async def close(self):
        if self.exchange:
            await self.exchange.close()
            print("🔌 [Trader] 거래소 연결이 안전하게 종료되었습니다.")