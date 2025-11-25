import sys
import os
import json
import joblib
import pandas as pd
import numpy as np
import ccxt.async_support as ccxt
from datetime import datetime
import asyncio
import aiosqlite
import zipfile 
import io      
import sqlite3 
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type

# 프로젝트 경로 설정
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import MODELS_DIR, DATA_DIR

# 중앙 집중식 피처 엔지니어링 함수
from alpha_layer.features import apply_features

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
        print(f"🆔 API KEY : {raw_api_key[:4]}****")
        print(f"🧪 테스트넷 : {IS_TESTNET}")
        print("="*40)
        
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
        
        self.backtest_dir = os.path.join(DATA_DIR, 'backtests')
        os.makedirs(self.backtest_dir, exist_ok=True)

        # DB 경로
        self.db_path = os.path.join(DATA_DIR, 'trade_state.db')
        
        # 모델 캐싱 저장소
        self.models = {} 

        # 전략 내부 상수
        self.EMA_PERIOD = 20
        self.COOLDOWN_BARS = 2
        
        # 동시성 제어
        self.trade_lock = asyncio.Lock() 
        self.cooldowns = {} 
        
        # 초기화 플래그
        self.is_initialized = False
        self.mode = 'OFF'
        self.state = {}

    async def initialize(self):
        """비동기 초기화: DB 생성 및 상태 로드"""
        if self.is_initialized: return
        await self._init_db()
        self.mode = await self._get_mode_from_db()
        self.state = await self._load_positions_from_db()
        await self.load_models() # 모델 메모리 로드
        self.is_initialized = True
        print(f"✅ [Trader] 초기화 완료 (Mode: {self.mode})")

    def get_conf(self, key, default=None):
        return self.config_manager.get(key, default)

    # -----------------------------------------------------------
    # [DB Section] Async SQLite Methods (aiosqlite)
    # -----------------------------------------------------------
    async def _init_db(self):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute('''
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
            await db.execute('''
                CREATE TABLE IF NOT EXISTS system_settings (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            ''')
            await db.execute('''
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
                    pnl REAL,
                    trade_id TEXT UNIQUE, 
                    trade_type TEXT DEFAULT 'TRADE'
                )
            ''')
            await db.execute('''
                CREATE TABLE IF NOT EXISTS backtest_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    symbol TEXT,
                    params TEXT,
                    roi REAL,
                    mdd REAL,
                    win_rate REAL,
                    trade_count INTEGER,
                    final_balance REAL,
                    csv_path TEXT
                )
            ''')
            
            # 마이그레이션
            try: await db.execute("ALTER TABLE positions ADD COLUMN entry_regime TEXT")
            except: pass
            try: await db.execute("ALTER TABLE positions ADD COLUMN stop_loss_price REAL DEFAULT 0.0")
            except: pass
            try: await db.execute("ALTER TABLE positions ADD COLUMN mode TEXT DEFAULT 'UNKNOWN'")
            except: pass
            try: await db.execute("ALTER TABLE trade_history ADD COLUMN trade_id TEXT")
            except: pass
            try: await db.execute("ALTER TABLE trade_history ADD COLUMN trade_type TEXT DEFAULT 'TRADE'")
            except: pass

            await db.execute("INSERT OR IGNORE INTO system_settings (key, value) VALUES ('mode', 'OFF')")
            await db.commit()

    async def _get_setting(self, key):
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute("SELECT value FROM system_settings WHERE key=?", (key,)) as cursor:
                res = await cursor.fetchone()
                return res[0] if res else None

    async def _set_setting(self, key, value):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("INSERT OR REPLACE INTO system_settings (key, value) VALUES (?, ?)", (key, str(value)))
            await db.commit()

    async def _get_mode_from_db(self):
        mode = await self._get_setting('mode')
        return mode if mode else 'OFF'

    async def _load_positions_from_db(self):
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute("""
                SELECT symbol, entry_price, highest_price, lowest_price, side, amount, entry_regime, stop_loss_price 
                FROM positions 
                WHERE mode = ?
            """, (self.mode,)) as cursor:
                rows = await cursor.fetchall()
        
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

    async def _upsert_position_to_db(self, symbol, entry_price, highest_price, lowest_price, side, amount, entry_regime, stop_loss_price):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("DELETE FROM positions WHERE symbol = ? AND mode = ?", (symbol, self.mode))
            await db.execute('''
                INSERT INTO positions (symbol, mode, entry_price, highest_price, lowest_price, side, amount, entry_regime, stop_loss_price)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (symbol, self.mode, entry_price, highest_price, lowest_price, side, amount, entry_regime, stop_loss_price))
            await db.commit()
        
        self.state[symbol] = {
            'entry_price': entry_price,
            'high': highest_price,
            'low': lowest_price,
            'side': side,
            'amount': amount,
            'entry_regime': entry_regime,
            'stop_loss_price': stop_loss_price
        }

    async def _delete_position_from_db(self, symbol):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("DELETE FROM positions WHERE symbol = ? AND mode = ?", (symbol, self.mode))
            await db.commit()
        
        if symbol in self.state:
            del self.state[symbol]

    # -----------------------------------------------------------
    # [거래 이력 저장]
    # -----------------------------------------------------------
    async def _save_trade_history(self, symbol, side, entry_price, exit_price, amount, real_fee=None, trade_type='TRADE'):
        if real_fee is not None:
            total_fee = real_fee
        else:
            commission_rate = self.get_conf('COMMISSION', 0.0005)
            if trade_type == 'TRADE':
                total_fee = (entry_price * amount * commission_rate) + (exit_price * amount * commission_rate)
            else:
                total_fee = 0.0 

        net_pnl = 0.0
        if trade_type == 'TRADE':
            if side == 'buy': 
                raw_pnl = (exit_price - entry_price) * amount
            else: 
                raw_pnl = (entry_price - exit_price) * amount
            net_pnl = raw_pnl - total_fee
        elif trade_type == 'FUNDING':
            net_pnl = -total_fee

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        trade_id_val = f"{trade_type}_{int(datetime.now().timestamp()*1000)}_{symbol}"
        
        async with aiosqlite.connect(self.db_path) as db:
            try:
                await db.execute('''
                    INSERT INTO trade_history (timestamp, symbol, mode, side, entry_price, exit_price, amount, fee, pnl, trade_id, trade_type)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (timestamp, symbol, self.mode, side, entry_price, exit_price, amount, total_fee, net_pnl, trade_id_val, trade_type))
            except sqlite3.IntegrityError:
                pass
            await db.commit()

        if self.mode == 'PAPER':
            await self._update_paper_balance(net_pnl)

        return net_pnl, total_fee

    async def _update_paper_balance(self, pnl):
        try:
            current_bal = await self.get_saved_paper_balance()
            new_bal = current_bal + pnl
            await self._set_setting('paper_balance', new_bal)
            print(f"💰 [Paper] 잔고 업데이트: ${current_bal:.2f} -> ${new_bal:.2f} (PnL: {pnl:+.4f})")
        except Exception as e:
            print(f"❌ 모의투자 잔고 업데이트 오류: {e}")

    async def get_trade_history(self, target_mode=None):
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            query = "SELECT * FROM trade_history"
            params = []
            if target_mode:
                query += " WHERE mode = ?"
                params.append(target_mode)
            query += " ORDER BY id ASC"
            
            async with db.execute(query, tuple(params)) as cursor:
                rows = await cursor.fetchall()
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
            if t['trade_type'] == 'TRADE':
                if pnl > 0: win_count += 1
                if pnl > max_pnl: max_pnl = pnl
            equity_curve.append({"time": t['timestamp'], "value": cumulative_pnl})

        normal_trades = [t for t in trades if t['trade_type'] == 'TRADE']
        total_trades = len(normal_trades)
        win_rate = round((win_count / total_trades * 100), 1) if total_trades > 0 else 0.0
        
        return {
            "summary": {"total_pnl": round(total_pnl, 2), "win_rate": win_rate, "total_trades": total_trades, "best_trade": round(max_pnl, 2)},
            "equity_curve": equity_curve,
            "trades": trades[::-1]
        }

    # -----------------------------------------------------------
    # [Model Management] Caching & Loading
    # -----------------------------------------------------------
    async def load_models(self):
        print("🧠 AI 모델 메모리 로딩 시작...")
        loaded_count = 0
        if not os.path.exists(MODELS_DIR):
            print("⚠️ 모델 디렉토리가 없습니다.")
            return

        for filename in os.listdir(MODELS_DIR):
            if filename.endswith(".pkl"):
                symbol_key = filename.replace("hmm_", "").replace(".pkl", "")
                path = os.path.join(MODELS_DIR, filename)
                try:
                    model = joblib.load(path)
                    self.models[symbol_key] = model
                    loaded_count += 1
                except Exception as e:
                    print(f"❌ 모델 로드 실패 ({filename}): {e}")
        
        print(f"✅ 총 {loaded_count}개 모델 로드 완료.")

    def get_hmm_regime(self, df, symbol):
        clean_symbol = symbol.replace('/', '')
        model = self.models.get(clean_symbol)
        if model is None: return None, None

        try:
            X = df[['Log_Returns_Scaled', 'Range_Vol_Scaled', 'RSI_14_Scaled', 'OBV_Scaled']].values
            hidden_states = model.predict(X)
            stats = pd.DataFrame(X, columns=['Ret', 'Vol', 'RSI', 'OBV'])
            stats['Regime'] = hidden_states
            regime_means = stats.groupby('Regime')['Ret'].mean()
            bear_id = regime_means.idxmin()
            bull_id = regime_means.idxmax()
            sideways_id = list(set(range(model.n_components)) - {bull_id, bear_id})[0] if model.n_components > 2 else None
            return hidden_states[-1], {'bull': bull_id, 'bear': bear_id, 'sideways': sideways_id}
        except Exception as e:
            print(f"⚠️ Regime detection failed for {symbol}: {e}")
            return None, None

    async def fetch_data_and_features(self, symbol):
        try:
            timeframe = self.get_conf('TIMEFRAME', '1h')
            limit = self.get_conf('CANDLE_LIMIT', 300)
            candles = await self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            if not candles: return None
            
            df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df.set_index('timestamp', inplace=True)
            df = apply_features(df)
            return df
        except Exception as e:
            print(f"❌ Data Fetch Error ({symbol}): {e}")
            return None

    # -----------------------------------------------------------
    # [Execution] Async Orders
    # -----------------------------------------------------------
    @retry(stop=stop_after_attempt(3), wait=wait_fixed(1), retry=retry_if_exception_type((ccxt.NetworkError, ccxt.RequestTimeout, ccxt.ExchangeError)))
    async def _create_order_with_retry(self, symbol, type, side, amount, price=None, params={}):
        if type == 'market':
            return await self.exchange.create_market_order(symbol, side, amount, params)
        else:
            return await self.exchange.create_limit_order(symbol, side, amount, price, params)

    async def execute_order(self, symbol, side, amount, reduce_only=False):
        if self.mode == 'OFF': return None
        try:
            ticker = await self.exchange.fetch_ticker(symbol)
            current_price = ticker['last']
        except: return None

        if self.mode == 'PAPER':
            slippage = self.get_conf('SLIPPAGE_PCT', 0.0002)
            exec_price = current_price * (1 + slippage) if side == 'buy' else current_price * (1 - slippage)
            sim_id = f"SIM_{int(datetime.now().timestamp()*1000)}"
            await self.notification.log(f"🧪 [모의] {symbol} {side} ({exec_price:.4f})", level="PAPER")
            return exec_price, sim_id

        if self.mode == 'REAL':
            try:
                params = {'reduceOnly': True} if reduce_only else {}
                order = await self._create_order_with_retry(symbol, 'market', side, amount, params=params)
                exec_price = order.get('average', current_price) 
                return exec_price, order['id']
            except Exception as e:
                await self.notification.log(f"🚨 [실매매 실패] {symbol} {side}: {e}", level="ERROR")
                return None, None

    async def _fetch_real_commission(self, symbol, order_id):
        if self.mode != 'REAL': return 0.0
        try:
            trades = await self.exchange.fetch_my_trades(symbol, limit=10)
            total_fee = 0.0
            for t in trades:
                if str(t['info']['orderId']) == str(order_id):
                    if 'fee' in t:
                        cost = float(t['fee']['cost'])
                        currency = t['fee']['currency']
                        if currency == 'USDT': total_fee += cost
            return total_fee
        except: return 0.0

    # -----------------------------------------------------------
    # [Main Logic Loop] (Look-Ahead Bias Fix Applied)
    # -----------------------------------------------------------
    async def run_logic(self):
        if not self.is_initialized: await self.initialize()

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
                if df is None or len(df) < 30: continue
                
                # ---------------------------------------------------
                # [중요] Look-Ahead Bias(Repainting) 방지 로직 분리
                # ---------------------------------------------------
                
                # 1. 실행/감시용 (Real-time)
                # 현재 진행 중인 캔들의 가격. 손절/익절 감시 및 트레일링 스탑에 사용
                current_price = df['close'].iloc[-1] 
                
                # 2. 신호 판단용 (Confirmed)
                # 직전 마감된 캔들의 데이터. 진입/청산 신호 판단에 사용
                prev_close = df['close'].iloc[-2]
                prev_rsi = df['RSI_14'].iloc[-2]
                prev_ema = df[f'EMA_{self.EMA_PERIOD}'].iloc[-2]
                prev_atr = df['ATRr_14'].iloc[-2]

                # 3. Regime 판단 (Confirmed)
                # 마지막 진행 중인 캔들(iloc[-1])을 제외하고 전달하여
                # '확정된' 캔들 기준의 국면을 판단함.
                regime, r_map = self.get_hmm_regime(df.iloc[:-1], symbol)
                if regime is None: continue
                
                has_position = symbol in self.state
                
                # --- 청산 로직 ---
                if has_position:
                    pos_data = self.state[symbol]
                    entry_price = pos_data['entry_price']
                    stop_loss = pos_data.get('stop_loss_price', 0)
                    high_price = pos_data.get('high', entry_price)
                    low_price = pos_data.get('low', entry_price)
                    side = pos_data['side']
                    amount = pos_data['amount']
                    entry_regime = pos_data.get('entry_regime', 'unknown')

                    stop_atr = self.get_conf('STOP_LOSS_ATR', 2.0)
                    trail_trigger = prev_atr * self.get_conf('TRAIL_TRIGGER_ATR', 2.0)
                    trail_dist = prev_atr * self.get_conf('TRAIL_DIST_ATR', 2.0)
                    
                    if stop_loss == 0:
                        # 초기 스탑로스 계산 시에도 prev_atr 사용 권장
                        stop_loss = entry_price - (prev_atr * stop_atr) if side == 'buy' else entry_price + (prev_atr * stop_atr)

                    should_close = False
                    reason = ""
                    
                    # 1. 하드 스탑 & 트레일링 스탑 실행 (실시간 가격 기준)
                    # 손절은 즉각 반응해야 하므로 current_price 사용
                    if side == 'buy' and current_price < stop_loss: should_close = True; reason = "StopLoss"
                    elif side == 'sell' and current_price > stop_loss: should_close = True; reason = "StopLoss"
                    
                    # 2. 트레일링 스탑 '업데이트' (실시간 고가/저가 기준)
                    if not should_close:
                        updated = False
                        if side == 'buy':
                            if current_price > high_price: 
                                high_price = current_price; updated = True
                            if high_price >= entry_price + trail_trigger:
                                new_stop = high_price - trail_dist
                                if new_stop > stop_loss: stop_loss = new_stop; updated = True
                        else: # sell
                            if current_price < low_price: 
                                low_price = current_price; updated = True
                            if low_price <= entry_price - trail_trigger:
                                new_stop = low_price + trail_dist
                                if new_stop < stop_loss: stop_loss = new_stop; updated = True
                        
                        if updated:
                            await self._upsert_position_to_db(symbol, entry_price, high_price, low_price, side, amount, entry_regime, stop_loss)

                    # 3. 국면 전환 및 전략적 청산 (확정된 신호 기준)
                    if not should_close:
                        if (entry_regime == 'bull' and regime == r_map['bear']) or \
                           (entry_regime == 'bear' and regime == r_map['bull']):
                             should_close = True; reason = "RegimeChange"

                    if should_close:
                        close_side = 'sell' if side == 'buy' else 'buy'
                        # 청산 주문은 현재가 기준으로 실행
                        exec_price, order_id = await self.execute_order(symbol, close_side, amount, reduce_only=True)
                        
                        if exec_price:
                            real_fee = await self._fetch_real_commission(symbol, order_id)
                            pnl, _ = await self._save_trade_history(symbol, side, entry_price, exec_price, amount, real_fee=real_fee)
                            await self.notification.log(f"💰 [{symbol}] 익절/손절 ({reason}) PnL: ${pnl:.2f}", level="INFO")
                            await self._delete_position_from_db(symbol)
                            self.cooldowns[symbol] = self.COOLDOWN_BARS
                        continue

                # --- 진입 로직 ---
                else:
                    if current_pos_count >= self.get_conf('MAX_OPEN_POSITIONS', 3): continue
                    if symbol in self.cooldowns: continue
                    
                    rsi_buy_low = self.get_conf('RSI_BUY_LOWER', 30)
                    rsi_buy_high = self.get_conf('RSI_BUY_UPPER', 65)
                    rsi_sell_low = self.get_conf('RSI_SELL_LOWER', 20)
                    rsi_sell_high = self.get_conf('RSI_SELL_UPPER', 75)

                    signal = None
                    target_regime = None

                    # 진입 판단: 확정된 캔들(prev_) 사용
                    if regime == r_map['bull']:
                        if prev_close > prev_ema and rsi_buy_low < prev_rsi < rsi_buy_high:
                            signal = 'buy'; target_regime = 'bull'
                    elif regime == r_map['bear']:
                        if prev_close < prev_ema and rsi_sell_low < prev_rsi < rsi_sell_high:
                            signal = 'sell'; target_regime = 'bear'
                    
                    if signal:
                        leverage = self.get_conf('LEVERAGE', 1.0)
                        try: await self.exchange.set_leverage(leverage, symbol)
                        except: pass
                        
                        balance = total_equity
                        max_pos = self.get_conf('MAX_OPEN_POSITIONS', 3)
                        allocation = (balance / max_pos) * 0.95
                        # 수량 계산은 현재가(current_price) 기준이 정확함
                        qty = (allocation * leverage) / current_price
                        
                        exec_price, order_id = await self.execute_order(symbol, signal, qty)
                        if exec_price:
                            stop_atr = self.get_conf('STOP_LOSS_ATR', 2.0)
                            # 초기 손절폭은 변동성이 확정된 prev_atr 사용
                            sl_dist = prev_atr * stop_atr
                            sl_price = exec_price - sl_dist if signal == 'buy' else exec_price + sl_dist
                            
                            await self._upsert_position_to_db(symbol, exec_price, exec_price, exec_price, signal, qty, target_regime, sl_price)
                            await self.notification.log(f"🚀 [{symbol}] {signal.upper()} 진입 @ {exec_price}", level="INFO")

            return "✅ 매매 로직 실행 완료"

    # -----------------------------------------------------------
    # [Utils] Sync Funding Loop & Backtest Ops
    # -----------------------------------------------------------
    async def sync_funding_fee_loop(self):
        while True:
            try:
                if self.mode == 'PAPER':
                    now = datetime.utcnow()
                    if now.hour in [0, 8, 16] and now.minute < 5:
                        for sym, pos in list(self.state.items()):
                            val = pos['amount'] * pos['entry_price']
                            cost = val * 0.0001
                            await self._save_trade_history(sym, 'FUNDING', 0, 0, 0, real_fee=cost, trade_type='FUNDING')
                        await asyncio.sleep(3600)
                await asyncio.sleep(60)
            except Exception as e:
                print(f"Funding Sync Error: {e}")
                await asyncio.sleep(60)

    # [NEW] 백테스트 결과 데이터 조회 (CSV/ZIP) - 다중 데이터셋 지원
    async def get_backtest_result_data(self, record_id):
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute("SELECT csv_path, symbol FROM backtest_history WHERE id=?", (record_id,)) as cursor:
                row = await cursor.fetchone()
        
        if not row: return None
        csv_path, symbol = row
        
        if not csv_path or not os.path.exists(csv_path):
            return None

        try:
            results = []
            
            # 1. Batch Backtest (Zip) - Portfolio + Individual Coins
            if csv_path.endswith('.zip'):
                with zipfile.ZipFile(csv_path, 'r') as z:
                    # 1-1. Portfolio Summary (Total)
                    summary_file = next((n for n in z.namelist() if 'Portfolio_Summary.csv' in n), None)
                    if summary_file:
                        with z.open(summary_file) as f:
                            df = pd.read_csv(f)
                            data = []
                            for _, row in df.iterrows():
                                data.append({"time": str(row['Date']), "value": float(row['Total_Equity'])})
                            results.append({"label": "Total Portfolio", "data": data})
                    
                    # 1-2. Individual Coins
                    for filename in z.namelist():
                        if filename == summary_file: continue
                        if not filename.endswith('.csv'): continue
                        
                        # 파일명 파싱 (Format: BT_{timestamp}_{symbol}.csv)
                        label = filename.replace('.csv', '')
                        parts = label.split('_')
                        if len(parts) >= 3 and parts[0] == 'BT':
                            symbol_part = "_".join(parts[3:]) 
                            if symbol_part:
                                label = symbol_part
                        
                        with z.open(filename) as f:
                            df = pd.read_csv(f)
                            val_col = 'Portfolio_Value'
                            if val_col not in df.columns:
                                val_col = df.columns[-1] 
                            
                            coin_data = []
                            for _, row in df.iterrows():
                                coin_data.append({"time": str(row['Date']), "value": float(row[val_col])})
                            
                            results.append({"label": label, "data": coin_data})
                
                return results
            
            # 2. Single Backtest (CSV) - 기존 호환성 유지
            else:
                df = pd.read_csv(csv_path)
                data = []
                val_col = 'Portfolio_Value' if 'Portfolio_Value' in df.columns else df.columns[-1]
                for _, row in df.iterrows():
                    data.append({"time": str(row['Date']), "value": float(row[val_col])})
            
                return data
                
        except Exception as e:
            print(f"Error reading backtest file: {e}")
            return None

    async def save_backtest_result(self, result):
        async with aiosqlite.connect(self.db_path) as db:
            params = json.dumps(result.get('params', {}), ensure_ascii=False)
            ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            await db.execute('''
                INSERT INTO backtest_history (timestamp, symbol, params, roi, mdd, win_rate, trade_count, final_balance, csv_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (ts, result.get('symbol'), params, result.get('roi'), result.get('mdd'), result.get('win_rate'), 
                  result.get('trade_count'), result.get('final_balance'), result.get('csv_path')))
            await db.commit()

    async def get_backtest_history(self):
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("SELECT * FROM backtest_history ORDER BY id DESC") as cursor:
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]

    async def delete_backtest_record(self, rid):
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute("SELECT csv_path FROM backtest_history WHERE id=?", (rid,)) as cur:
                row = await cur.fetchone()
                if row and row[0] and os.path.exists(row[0]):
                    try: os.remove(row[0])
                    except: pass
            await db.execute("DELETE FROM backtest_history WHERE id=?", (rid,))
            await db.commit()

    async def get_saved_paper_balance(self):
        val = await self._get_setting('paper_balance')
        return float(val) if val else 10000.0

    async def reset_paper_balance(self, amount):
        await self._set_setting('paper_balance', amount)
        return amount

    async def set_mode(self, new_mode):
        if new_mode not in ['OFF', 'REAL', 'PAPER']: return "Invalid Mode"
        if not self.is_initialized: await self.initialize()
        self.mode = new_mode
        await self._set_setting('mode', new_mode)
        self.state = await self._load_positions_from_db()
        if new_mode != 'OFF': await self.load_models()
        msg = f"✅ 모드 변경: {new_mode}"
        await self.notification.log(msg, level="SYSTEM")
        return msg

    async def get_balance(self):
        if self.mode == 'PAPER':
            bal = await self.get_saved_paper_balance()
            return bal, bal
        try:
            bal = await self.exchange.fetch_balance()
            return bal['free']['USDT'], bal['total']['USDT']
        except: return 0.0, 0.0

    async def get_positions(self):
        if self.mode != 'REAL':
            return [
                {'symbol': s, 'side': p['side'], 'amount': p['amount'], 
                 'entryPrice': p['entry_price'], 'unrealizedPnl': 0, 
                 'leverage': self.get_conf('LEVERAGE', 1), 'mode': self.mode}
                for s, p in self.state.items()
            ]
        return []
    
    async def safe_force_close(self, symbol):
        symbol = symbol.strip()
        async with self.trade_lock:
            try: await self.exchange.cancel_all_orders(symbol)
            except: pass
            if symbol not in self.state: return f"⚠️ {symbol} 청산 실패: 포지션 없음"
            data = self.state[symbol]
            side = 'sell' if data['side'] == 'buy' else 'buy'
            amount = data['amount']
            res = await self.execute_order(symbol, side, amount, reduce_only=True)
            if not res: return f"❌ {symbol} 청산 주문 실패"
            exec_price, order_id = res
            if exec_price:
                real_fee = await self._fetch_real_commission(symbol, order_id)
                pnl, _ = await self._save_trade_history(symbol, data['side'], data['entry_price'], exec_price, amount, real_fee=real_fee, trade_type='TRADE')
                await self._delete_position_from_db(symbol)
                msg = f"✅ {symbol} 청산 완료. PnL: ${pnl:.2f}"
                await self.notification.log(msg, level="INFO")
                return msg
            else: return f"❌ {symbol} 청산 실패"

    async def close_all_positions(self):
        """모든 포지션을 강제 청산하는 메서드"""
        if not self.state:
            return "⚠️ 청산할 포지션이 없습니다."

        logs = []
        # 딕셔너리 크기 변경 오류 방지를 위해 키를 리스트로 복사하여 순회
        target_symbols = list(self.state.keys())
        
        for symbol in target_symbols:
            msg = await self.safe_force_close(symbol)
            logs.append(msg)
        
        return "\n".join(logs)

    async def close(self):
        await self.exchange.close()