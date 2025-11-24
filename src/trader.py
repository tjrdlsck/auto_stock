import sys
import os
import json
import joblib
import pandas as pd
import pandas_ta as ta
import numpy as np
import ccxt.async_support as ccxt
from datetime import datetime, timedelta
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
        
        # [NEW] 백테스트 결과 저장 디렉토리 생성
        self.backtest_dir = os.path.join(DATA_DIR, 'backtests')
        os.makedirs(self.backtest_dir, exist_ok=True)

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
    # [DB Section] SQLite Helper Methods (Updated)
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
        
        # 2. 시스템 설정 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_settings (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        ''')

        # 3. 거래 이력 테이블 (컬럼 추가: trade_id, trade_type)
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
                pnl REAL,
                trade_id TEXT UNIQUE, 
                trade_type TEXT DEFAULT 'TRADE'
            )
        ''')
        
        # 4. 백테스트 이력 테이블
        cursor.execute('''
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

        # 마이그레이션 (활성 포지션 테이블)
        cursor.execute("PRAGMA table_info(positions)")
        pos_columns = [info[1] for info in cursor.fetchall()]
        if 'entry_regime' not in pos_columns:
            cursor.execute("ALTER TABLE positions ADD COLUMN entry_regime TEXT")
        if 'stop_loss_price' not in pos_columns:
            cursor.execute("ALTER TABLE positions ADD COLUMN stop_loss_price REAL DEFAULT 0.0")
        if 'mode' not in pos_columns:
            cursor.execute("ALTER TABLE positions ADD COLUMN mode TEXT DEFAULT 'UNKNOWN'")

        # [NEW] 마이그레이션 (거래 이력 테이블)
        cursor.execute("PRAGMA table_info(trade_history)")
        hist_columns = [info[1] for info in cursor.fetchall()]
        if 'trade_id' not in hist_columns:
            cursor.execute("ALTER TABLE trade_history ADD COLUMN trade_id TEXT")
        if 'trade_type' not in hist_columns:
            cursor.execute("ALTER TABLE trade_history ADD COLUMN trade_type TEXT DEFAULT 'TRADE'")

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
    # [중요] 거래 이력 저장 (Updated: 실제 수수료 및 타입 지원)
    # -----------------------------------------------------------
    def _save_trade_history(self, symbol, side, entry_price, exit_price, amount, real_fee=None, trade_type='TRADE'):
        """청산 또는 펀딩비 발생 시 거래 이력 저장 및 모의투자 잔고 업데이트"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 수수료 계산 (실제 값이 없으면 추정치 사용)
        if real_fee is not None:
            total_fee = real_fee
        else:
            commission_rate = self.get_conf('COMMISSION', 0.0005)
            # 일반 매매일 경우 진입+청산 수수료 합산 추정
            if trade_type == 'TRADE':
                total_fee = (entry_price * amount * commission_rate) + (exit_price * amount * commission_rate)
            else:
                total_fee = 0.0

        # PnL 계산
        net_pnl = 0.0
        if trade_type == 'TRADE':
            if side == 'buy': # Long 청산 (매도)
                raw_pnl = (exit_price - entry_price) * amount
            else: # Short 청산 (매수)
                raw_pnl = (entry_price - exit_price) * amount
            net_pnl = raw_pnl - total_fee
        elif trade_type == 'FUNDING':
            # 펀딩비는 비용(음수 PnL)으로 처리 (real_fee가 양수면 지출)
            net_pnl = -total_fee

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # trade_id 생성 (REAL 모드가 아니거나 펀딩비 시뮬레이션일 경우 랜덤)
        trade_id_val = f"{trade_type}_{int(datetime.now().timestamp()*1000)}_{symbol}"
        
        # 1. 거래 이력 저장
        try:
            cursor.execute('''
                INSERT INTO trade_history (timestamp, symbol, mode, side, entry_price, exit_price, amount, fee, pnl, trade_id, trade_type)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (timestamp, symbol, self.mode, side, entry_price, exit_price, amount, total_fee, net_pnl, trade_id_val, trade_type))
        except sqlite3.IntegrityError:
            pass # 중복 ID 무시
        
        # 2. 모의투자 잔고 업데이트
        if self.mode == 'PAPER':
            try:
                cursor.execute("SELECT value FROM system_settings WHERE key='paper_balance'")
                res = cursor.fetchone()
                current_bal = float(res[0]) if res else 10000.0
                
                new_bal = current_bal + net_pnl
                cursor.execute("INSERT OR REPLACE INTO system_settings (key, value) VALUES ('paper_balance', ?)", (str(new_bal),))
                print(f"💰 [Paper] 잔고 업데이트 ({trade_type}): ${current_bal:.2f} -> ${new_bal:.2f} (PnL: {net_pnl:+.4f})")
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
            # 펀딩비는 승률 계산에서 제외하거나 포함 정책 결정 필요 (여기선 제외)
            if t['trade_type'] == 'TRADE':
                if pnl > 0: win_count += 1
                if pnl > max_pnl: max_pnl = pnl
            equity_curve.append({"time": t['timestamp'], "value": cumulative_pnl})

        # 일반 매매 횟수만 카운트
        normal_trades = [t for t in trades if t['trade_type'] == 'TRADE']
        total_trades = len(normal_trades)
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

    # -----------------------------------------------------------
    # [NEW] 실제 수수료 조회 헬퍼 (Phase 1 기능)
    # -----------------------------------------------------------
    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    async def _fetch_real_commission(self, symbol, order_id):
        """체결된 주문의 실제 수수료를 USDT 단위로 조회"""
        if self.mode != 'REAL': return 0.0

        # 최근 체결 내역 조회
        trades = await self.exchange.fetch_my_trades(symbol, limit=30)
        
        total_fee_usdt = 0.0
        found_trades = False

        for trade in trades:
            if str(trade['info']['orderId']) == str(order_id):
                found_trades = True
                if 'fee' in trade and trade['fee']:
                    fee_cost = float(trade['fee']['cost'])
                    fee_currency = trade['fee']['currency']

                    # BNB 수수료인 경우 USDT 환산
                    if fee_currency == 'BNB':
                        bnb_ticker = await self.exchange.fetch_ticker('BNB/USDT')
                        bnb_price = bnb_ticker['last']
                        total_fee_usdt += (fee_cost * bnb_price)
                    elif fee_currency == 'USDT':
                        total_fee_usdt += fee_cost
                    else:
                        pass # 기타 코인은 무시 (혹은 추가 구현)
        
        if not found_trades:
            # 아직 데이터가 안 들어왔을 수 있음 -> Retry 트리거
            raise ValueError(f"Order ID {order_id} not found in trades yet.")
            
        return total_fee_usdt

    # -----------------------------------------------------------
    # [NEW] 펀딩비 동기화 루프 (Phase 1 기능)
    # -----------------------------------------------------------
    async def sync_funding_fee_loop(self):
        """주기적으로 펀딩비 내역을 확인하고 동기화"""
        print("⏳ [Funding] 펀딩비 동기화 루프 시작")
        while True:
            try:
                # 1. 모의투자 (PAPER) 시뮬레이션
                if self.mode == 'PAPER':
                    now = datetime.utcnow() # UTC 기준
                    # 00, 08, 16시 정각 근처 (0분~5분 사이)에만 실행
                    if now.hour in [0, 8, 16] and now.minute < 5:
                        for sym, pos in self.state.items():
                            # 간단하게: 진입가 * 수량 * 0.01%
                            # 더 정확하게 하려면 현재가(fetch_ticker)를 써야 하지만 시뮬레이션임.
                            pos_val = pos['amount'] * pos['entry_price']
                            funding_cost = pos_val * 0.0001 # 0.01%
                            
                            # 최근 1시간 내에 처리된 펀딩비가 있는지 DB 체크 로직 생략 (간소화)
                            # 대신 1시간 sleep으로 중복 방지
                            
                            self._save_trade_history(
                                symbol=sym, side='FUNDING', entry_price=0, exit_price=0, 
                                amount=0, real_fee=funding_cost, trade_type='FUNDING'
                            )
                            await self.notification.log(f"💸 [모의] 펀딩비 차감 ({sym}): ${funding_cost:.4f}", level="PAPER")
                        
                        # 중복 실행 방지를 위해 1시간 대기
                        await asyncio.sleep(3600)
                    else:
                        # 해당 시간이 아니면 1분 대기
                        await asyncio.sleep(60)

                # 2. 실전매매 (REAL) 동기화
                elif self.mode == 'REAL':
                    # TODO: fetch_income API 연동 (복잡도 때문에 추후 상세 구현 권장)
                    # 현재는 Placeholder
                    await asyncio.sleep(60)
                
                else:
                    await asyncio.sleep(60)

            except Exception as e:
                print(f"⚠️ Funding Sync Error: {e}")
                await asyncio.sleep(60)

    # -----------------------------------------------------------
    # [Backtest & Paper Balance]
    # -----------------------------------------------------------
    def save_backtest_result(self, result_dict):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        params_json = json.dumps(result_dict.get('params', {}), ensure_ascii=False)
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cursor.execute('''
            INSERT INTO backtest_history (timestamp, symbol, params, roi, mdd, win_rate, trade_count, final_balance, csv_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (timestamp, result_dict.get('symbol','Unknown'), params_json, result_dict.get('roi',0), result_dict.get('mdd',0), result_dict.get('win_rate',0), result_dict.get('trade_count',0), result_dict.get('final_balance',0), result_dict.get('csv_path','')))
        conn.commit()
        conn.close()

    def get_backtest_history(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM backtest_history ORDER BY id DESC")
        rows = cursor.fetchall()
        conn.close()
        history = []
        for row in rows:
            d = dict(row)
            try: d['params'] = json.loads(d['params'])
            except: d['params'] = {}
            history.append(d)
        return history

    def delete_backtest_record(self, record_id):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT csv_path FROM backtest_history WHERE id=?", (record_id,))
        row = cursor.fetchone()
        if row and row[0] and os.path.exists(row[0]):
            try: os.remove(row[0])
            except: pass
        cursor.execute("DELETE FROM backtest_history WHERE id=?", (record_id,))
        conn.commit()
        conn.close()
        return True

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
    # [Info Section - MISSING PART FIXED]
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
    # [Logic Helper]
    # -----------------------------------------------------------
    async def set_leverage(self, symbol, leverage):
        try: await self.exchange.set_leverage(leverage, symbol)
        except: pass 

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
            if now_utc < last_candle_ts + pd.to_timedelta(timeframe): candles = candles[:-1]
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
            for col in ['Log_Returns', 'Range_Vol', 'RSI_14', 'OBV']:
                rolling_mean = df[col].rolling(window=window).mean()
                rolling_std = df[col].rolling(window=window).std()
                df[f'{col}_Scaled'] = (df[col] - rolling_mean) / (rolling_std + 1e-8)
            return df.dropna()
        except: return None

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
        except: return None, None

    # -----------------------------------------------------------
    # [Execution Section] (Updated: Return Tuple)
    # -----------------------------------------------------------
    @retry(stop=stop_after_attempt(3), wait=wait_fixed(1), retry=retry_if_exception_type((ccxt.NetworkError, ccxt.RequestTimeout, ccxt.ExchangeError)))
    async def _create_market_order_with_retry(self, symbol, side, amount, params):
        return await self.exchange.create_market_order(symbol, side, amount, params)

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(1), retry=retry_if_exception_type((ccxt.NetworkError, ccxt.RequestTimeout, ccxt.ExchangeError)))
    async def _create_limit_order_with_retry(self, symbol, side, amount, price, params):
        return await self.exchange.create_limit_order(symbol, side, amount, price, params)

    async def _smart_execute_order(self, symbol, side, amount, reduce_only=False):
        """
        지정가 주문 시도 -> 타임아웃 -> 시장가 전환
        반환: (평균체결가, order_id)
        """
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
                    return order_status['average'], order_id
                await asyncio.sleep(1)

            # 타임아웃: 취소 후 잔량 시장가
            try:
                await self.exchange.cancel_order(order_id, symbol)
            except: pass # 이미 체결되었을 수 있음

            order_status = await self.exchange.fetch_order(order_id, symbol)
            filled = float(order_status.get('filled', 0.0))
            avg_fill_price = order_status.get('average')
            
            if filled >= amount:
                return avg_fill_price, order_id

            if force_market:
                remaining = amount - filled
                if remaining > 0:
                    market_order = await self._create_market_order_with_retry(symbol, side, remaining, params)
                    market_price = market_order['average']
                    m_order_id = market_order['id']
                    
                    # 가중 평균가 계산
                    total_cost = (market_price * remaining)
                    if filled > 0 and avg_fill_price:
                        total_cost += (avg_fill_price * filled)
                    
                    final_avg_price = total_cost / amount
                    return final_avg_price, m_order_id # 마지막 주문 ID 반환
                else:
                    return avg_fill_price, order_id
            
            # 시장가 전환 안 함
            return (avg_fill_price, order_id) if filled > 0 else (None, None)

        except Exception as e:
            await self.notification.log(f"❌ 주문 실행 에러 ({symbol}): {e}", level="ERROR")
            return None, None

    async def execute_order(self, symbol, side, amount, reduce_only=False):
        """
        주문 실행 (REAL/PAPER 공통 인터페이스)
        반환: (price, order_id) 또는 None
        """
        if self.mode == 'OFF': return None

        try:
            ticker = await self.exchange.fetch_ticker(symbol)
            current_price = ticker['last']
        except: return None

        msg_type = "청산" if reduce_only else "진입"

        # 1. 모의매매
        if self.mode == 'PAPER':
            slippage = self.get_conf('SLIPPAGE_PCT', 0.0002)
            exec_price = current_price * (1 + slippage) if side == 'buy' else current_price * (1 - slippage)
            
            # 모의 주문 ID 생성
            sim_order_id = f"SIM_{int(datetime.now().timestamp()*1000)}"
            log_msg = f"🧪 [모의] {symbol} {side.upper()} {msg_type} ({exec_price:.4f})"
            await self.notification.log(log_msg, level="PAPER")
            return exec_price, sim_order_id

        # 2. 실매매
        if self.mode == 'REAL':
            exec_price, order_id = await self._smart_execute_order(symbol, side, amount, reduce_only)
            if exec_price:
                log_msg = f"🚀 [실매매] {symbol} {side.upper()} {msg_type} ({exec_price})"
                await self.notification.log(log_msg, level="REAL")
                return exec_price, order_id
            else:
                await self.notification.log(f"🚨 [긴급] {symbol} {side} 주문 실패", level="ERROR")
                return None

    # -----------------------------------------------------------
    # [청산 로직 - Updated]
    # -----------------------------------------------------------
    async def safe_force_close(self, symbol):
        symbol = symbol.strip()
        
        async with self.trade_lock:
            try: await self.exchange.cancel_all_orders(symbol)
            except: pass
            
            if symbol not in self.state:
                return f"⚠️ {symbol} 청산 실패: 포지션 없음"

            data = self.state[symbol]
            side = 'sell' if data['side'] == 'buy' else 'buy'
            amount = data['amount']
            
            # execute_order 반환값 언패킹
            res = await self.execute_order(symbol, side, amount, reduce_only=True)
            if not res:
                return f"❌ {symbol} 청산 주문 실패"

            exec_price, order_id = res
            
            if exec_price:
                # 실제 수수료 조회 (REAL 모드만)
                real_fee = 0.0
                if self.mode == 'REAL' and order_id:
                    try:
                        real_fee = await self._fetch_real_commission(symbol, order_id)
                        await self.notification.log(f"🧾 실제 수수료: ${real_fee:.4f}", level="REAL")
                    except Exception as e:
                        await self.notification.log(f"⚠️ 수수료 조회 실패 (추정치 사용): {e}", level="WARN")

                # 이력 저장
                pnl, fee = self._save_trade_history(symbol, data['side'], data['entry_price'], exec_price, amount, real_fee=real_fee, trade_type='TRADE')
                
                self._delete_position_from_db(symbol)
                del self.state[symbol]
                
                msg = f"✅ {symbol} 청산 완료. PnL: ${pnl:.2f}"
                await self.notification.log(msg, level="INFO")
                return msg
            else:
                return f"❌ {symbol} 청산 실패 (체결가 없음)"
    
    async def close_all_positions(self):
        if not self.state: return "ℹ️ 청산할 포지션이 없습니다."
        symbols = list(self.state.keys())
        await self.notification.log(f"🚨 일괄 청산 시작 ({len(symbols)}개)", level="SYSTEM")
        results = []
        for symbol in symbols:
            res = await self.safe_force_close(symbol)
            results.append(res)
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

                    # StopLoss & Trailing Stop & Regime Change Logic (기존 유지)
                    if pos_side == 'buy' and curr_price < stop_loss_price: should_close = True; exit_reason = "StopLoss"
                    elif pos_side == 'sell' and curr_price > stop_loss_price: should_close = True; exit_reason = "StopLoss"
                    
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

                    if not should_close:
                        if (entry_regime == 'bull' and regime == r_map['bear']) or \
                           (entry_regime == 'bear' and regime == r_map['bull']):
                            should_close = True; exit_reason = "RegimeChange"

                    if should_close:
                        await self.notification.log(f"🔥 [{symbol}] 청산 신호 ({exit_reason})", level="INFO")
                        close_side = 'sell' if pos_side == 'buy' else 'buy'
                        
                        # [Modified] 언패킹 처리
                        res = await self.execute_order(symbol, close_side, pos_amt, reduce_only=True)
                        if res:
                            exec_price, order_id = res
                            real_fee = 0.0
                            if self.mode == 'REAL' and order_id:
                                try: real_fee = await self._fetch_real_commission(symbol, order_id)
                                except: pass
                            
                            pnl, fee = self._save_trade_history(symbol, pos_side, entry_price, exec_price, pos_amt, real_fee=real_fee)
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
                        
                        # [Modified] 언패킹 처리
                        res = await self.execute_order(symbol, entry_signal, qty_contract)
                        if res:
                            exec_price, order_id = res
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