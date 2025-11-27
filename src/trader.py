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
        self.model_metas = {}

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
            # 포지션 테이블 (entry_time 추가됨)
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
                    stop_loss_price REAL DEFAULT 0.0,
                    entry_time TEXT
                )
            ''')
            # 시스템 설정 테이블
            await db.execute('''
                CREATE TABLE IF NOT EXISTS system_settings (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            ''')
            # 거래 기록 테이블 (분석 컬럼 대거 추가)
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
                    trade_type TEXT DEFAULT 'TRADE',
                    strategy TEXT,
                    entry_regime TEXT,
                    exit_reason TEXT,
                    duration INTEGER,
                    roi REAL
                )
            ''')
            # 백테스트 기록 테이블
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
            
            # [DB 마이그레이션] 기존 DB 호환성 유지를 위한 컬럼 추가
            try: await db.execute("ALTER TABLE positions ADD COLUMN entry_regime TEXT")
            except: pass
            try: await db.execute("ALTER TABLE positions ADD COLUMN stop_loss_price REAL DEFAULT 0.0")
            except: pass
            try: await db.execute("ALTER TABLE positions ADD COLUMN mode TEXT DEFAULT 'UNKNOWN'")
            except: pass
            try: await db.execute("ALTER TABLE positions ADD COLUMN entry_time TEXT")
            except: pass
            
            try: await db.execute("ALTER TABLE trade_history ADD COLUMN trade_id TEXT")
            except: pass
            try: await db.execute("ALTER TABLE trade_history ADD COLUMN trade_type TEXT DEFAULT 'TRADE'")
            except: pass
            
            # 신규 분석 컬럼 추가
            try: await db.execute("ALTER TABLE trade_history ADD COLUMN strategy TEXT")
            except: pass
            try: await db.execute("ALTER TABLE trade_history ADD COLUMN entry_regime TEXT")
            except: pass
            try: await db.execute("ALTER TABLE trade_history ADD COLUMN exit_reason TEXT")
            except: pass
            try: await db.execute("ALTER TABLE trade_history ADD COLUMN duration INTEGER")
            except: pass
            try: await db.execute("ALTER TABLE trade_history ADD COLUMN roi REAL")
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
            # entry_time 컬럼 추가 조회
            async with db.execute("""
                SELECT symbol, entry_price, highest_price, lowest_price, side, amount, entry_regime, stop_loss_price, entry_time 
                FROM positions 
                WHERE mode = ?
            """, (self.mode,)) as cursor:
                rows = await cursor.fetchall()
        
        state = {}
        for row in rows:
            # Unpacking에 entry_time 추가
            symbol, entry_price, highest_price, lowest_price, side, amount, entry_regime, stop_loss_price, entry_time = row
            state[symbol] = {
                'entry_price': entry_price,
                'high': highest_price,
                'low': lowest_price,
                'side': side,
                'amount': amount,
                'entry_regime': entry_regime,
                'stop_loss_price': stop_loss_price,
                'entry_time': entry_time  # 상태 딕셔너리에 저장
            }
        return state

    async def _upsert_position_to_db(self, symbol, entry_price, highest_price, lowest_price, side, amount, entry_regime, stop_loss_price, entry_time=None):
        # entry_time 인자 추가 및 저장 로직 반영
        if entry_time is None:
            # 기존에 시간이 있으면 유지, 없으면 현재 시간 (새로 진입 시)
            if symbol in self.state and 'entry_time' in self.state[symbol]:
                entry_time = self.state[symbol]['entry_time']
            else:
                entry_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("DELETE FROM positions WHERE symbol = ? AND mode = ?", (symbol, self.mode))
            await db.execute('''
                INSERT INTO positions (symbol, mode, entry_price, highest_price, lowest_price, side, amount, entry_regime, stop_loss_price, entry_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (symbol, self.mode, entry_price, highest_price, lowest_price, side, amount, entry_regime, stop_loss_price, entry_time))
            await db.commit()
        
        self.state[symbol] = {
            'entry_price': entry_price,
            'high': highest_price,
            'low': lowest_price,
            'side': side,
            'amount': amount,
            'entry_regime': entry_regime,
            'stop_loss_price': stop_loss_price,
            'entry_time': entry_time
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
    async def _save_trade_history(self, symbol, side, entry_price, exit_price, amount, real_fee=None, trade_type='TRADE', 
                                  strategy=None, entry_regime=None, exit_reason=None, duration=0):
        # 1. 수수료 계산
        if real_fee is not None:
            total_fee = real_fee
        else:
            commission_rate = self.get_conf('COMMISSION', 0.0005)
            if trade_type == 'TRADE':
                total_fee = (entry_price * amount * commission_rate) + (exit_price * amount * commission_rate)
            else:
                total_fee = 0.0 

        net_pnl = 0.0
        roi = 0.0
        
        # 2. PnL 및 ROI 계산
        if trade_type == 'TRADE':
            if side == 'buy': 
                raw_pnl = (exit_price - entry_price) * amount
            else: 
                raw_pnl = (entry_price - exit_price) * amount
            
            net_pnl = raw_pnl - total_fee
            
            # [FIX] 안전한 ROI 계산 (ZeroDivisionError 방지)
            try:
                leverage = float(self.get_conf('LEVERAGE', 1.0))
                if leverage <= 0: leverage = 1.0 # 레버리지 0 방지
                
                # 투입 증거금 = (진입가 * 수량) / 레버리지
                invested_margin = (entry_price * amount) / leverage
                
                if invested_margin > 0.00000001: # 0에 가까운 수 방지
                    roi = (net_pnl / invested_margin) * 100
                else:
                    roi = 0.0
            except Exception as e:
                print(f"⚠️ ROI Calculation Error: {e}")
                roi = 0.0

        elif trade_type == 'FUNDING':
            net_pnl = -total_fee

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        trade_id_val = f"{trade_type}_{int(datetime.now().timestamp()*1000)}_{symbol}"
        
        # 3. DB 저장
        async with aiosqlite.connect(self.db_path) as db:
            try:
                await db.execute('''
                    INSERT INTO trade_history (
                        timestamp, symbol, mode, side, entry_price, exit_price, amount, fee, pnl, 
                        trade_id, trade_type, strategy, entry_regime, exit_reason, duration, roi
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (timestamp, symbol, self.mode, side, entry_price, exit_price, amount, total_fee, net_pnl, 
                      trade_id_val, trade_type, strategy, entry_regime, exit_reason, duration, roi))
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
    
    async def export_history_csv(self, target_mode):
        """
        특정 모드의 거래 기록을 CSV 포맷 문자열로 내보냅니다.
        분석을 위해 가능한 모든 상세 데이터를 포함합니다.
        """
        import pandas as pd
        import io
        
        async with aiosqlite.connect(self.db_path) as db:
            # pandas.read_sql을 쓰려면 동기 커넥션이 필요하므로, aiosqlite 대신 직접 fetch 후 변환
            db.row_factory = aiosqlite.Row
            async with db.execute("SELECT * FROM trade_history WHERE mode = ? ORDER BY id DESC", (target_mode,)) as cursor:
                rows = await cursor.fetchall()
                # 컬럼명 가져오기
                columns = [description[0] for description in cursor.description]
                
        if not rows:
            return ""

        # 데이터프레임 생성
        data = [dict(row) for row in rows]
        df = pd.DataFrame(data, columns=columns)
        
        # [FIX] 분석용 컬럼 순서 재정렬 및 확장 (누락 방지)
        target_cols = [
            'timestamp', 'symbol', 'side', 'type', 'strategy', 
            'entry_regime', 'exit_reason', 'duration',
            'entry_price', 'exit_price', 'amount', 'leverage', # leverage는 DB에 없으면 계산 필요하지만 일단 제외
            'pnl', 'roi', 'fee', 'trade_id'
        ]
        
        # 실제 DB에 존재하는 컬럼만 교집합으로 선택하여 순서 정렬
        final_cols = [c for c in target_cols if c in df.columns]
        
        # 만약 target_cols에 없는 나머지 컬럼이 있다면 뒤에 붙임 (데이터 손실 방지)
        remaining_cols = [c for c in df.columns if c not in final_cols and c not in ['id', 'mode']]
        final_cols.extend(remaining_cols)
        
        df = df[final_cols]

        # CSV 버퍼에 쓰기
        stream = io.StringIO()
        df.to_csv(stream, index=False)
        return stream.getvalue()
    
    # -----------------------------------------------------------
    # [Model Management] Caching & Loading
    # -----------------------------------------------------------
    async def load_models(self):
        print("🧠 AI 모델 및 기준표(Metadata) 메모리 로딩 시작...")
        loaded_count = 0
        if not os.path.exists(MODELS_DIR):
            print("⚠️ 모델 디렉토리가 없습니다.")
            return

        for filename in os.listdir(MODELS_DIR):
            # .pkl 파일만 찾음
            if filename.endswith(".pkl"):
                symbol_key = filename.replace("hmm_", "").replace(".pkl", "")
                
                # 파일 경로 설정
                model_path = os.path.join(MODELS_DIR, filename)
                # 메타데이터 파일명 추정 (Phase 1에서 저장한 규칙: hmm_{symbol}_meta.json)
                meta_filename = filename.replace(".pkl", "_meta.json")
                meta_path = os.path.join(MODELS_DIR, meta_filename)

                try:
                    # 1. 모델 로드
                    model = joblib.load(model_path)
                    self.models[symbol_key] = model
                    
                    # 2. 메타데이터(기준표) 로드 [NEW]
                    if os.path.exists(meta_path):
                        with open(meta_path, 'r', encoding='utf-8') as f:
                            self.model_metas[symbol_key] = json.load(f)
                    else:
                        # 메타파일이 없으면 None 처리 (구버전 호환용)
                        print(f"⚠️ [{symbol_key}] 기준표 파일(_meta.json)이 없습니다. (Fallback 모드 작동)")
                        self.model_metas[symbol_key] = None

                    loaded_count += 1
                except Exception as e:
                    print(f"❌ 모델/메타 로드 실패 ({filename}): {e}")
        
        print(f"✅ 총 {loaded_count}개 모델 세트 로드 완료.")

    def get_hmm_regime(self, df, symbol):
        clean_symbol = symbol.replace('/', '')
        model = self.models.get(clean_symbol)
        meta = self.model_metas.get(clean_symbol) # 로드된 기준표 가져오기
        
        if model is None: return None, None

        try:
            # 피처 데이터 추출
            X = df[['Log_Returns_Scaled', 'Range_Vol_Scaled', 'RSI_14_Scaled', 'OBV_Scaled']].values
            
            # HMM 예측 (0, 1, 2 중 하나의 숫자 반환)
            hidden_states = model.predict(X)
            current_state = int(hidden_states[-1]) # 가장 최근 캔들의 상태
            
            # -----------------------------------------------------------
            # [핵심 수정] 저장된 기준표(Meta)를 우선 사용 (절대평가)
            # -----------------------------------------------------------
            if meta is not None:
                # meta 구조: {'bull': 2, 'bear': 0, 'sideways': 1}
                # 반환값 1: 현재 상태 번호 (예: 0)
                # 반환값 2: 맵핑 딕셔너리 (run_logic에서 r_map['bull']과 비교함)
                return current_state, meta

            # -----------------------------------------------------------
            # [Fallback] 기준표가 없을 때만 기존 방식 사용 (상대평가)
            # (Phase 1을 실행하지 않았을 경우를 대비한 안전장치)
            # -----------------------------------------------------------
            stats = pd.DataFrame(X, columns=['Ret', 'Vol', 'RSI', 'OBV'])
            stats['Regime'] = hidden_states
            regime_means = stats.groupby('Regime')['Ret'].mean()
            
            bear_id = int(regime_means.idxmin())
            bull_id = int(regime_means.idxmax())
            
            # 남은 하나를 횡보(Sideways)로 정의
            all_ids = set(range(model.n_components))
            sideways_id = list(all_ids - {bull_id, bear_id})[0] if model.n_components > 2 else None
            
            fallback_map = {'bull': bull_id, 'bear': bear_id, 'sideways': sideways_id}
            return current_state, fallback_map

        except Exception as e:
            print(f"⚠️ Regime detection failed for {symbol}: {e}")
            return None, None

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
    async def execute_smart_order(self, symbol, side, amount, reduce_only=False):
        """
        [Phase 4 수정] Smart Execution (WAP 계산 + Dust 방지 + ID 리스트 반환)
        Returns: (weighted_avg_price, [order_ids], exec_type)
        """
        # 모의투자: 즉시 처리 (단일 ID 리스트 반환으로 통일)
        if self.mode == 'PAPER':
            slippage = self.get_conf('SLIPPAGE_PCT', 0.0002)
            ticker = await self.exchange.fetch_ticker(symbol)
            curr_price = ticker['last']
            exec_price = curr_price * (1 + slippage) if side == 'buy' else curr_price * (1 - slippage)
            sim_id = f"SIM_{int(datetime.now().timestamp()*1000)}"
            return exec_price, [sim_id], 'MARKET'

        if self.mode != 'REAL': return None, [], None

        # REAL 모드 설정
        priority = self.get_conf('ORDER_TYPE_PRIORITY', 'LIMIT')
        attempts = self.get_conf('LIMIT_ORDER_ATTEMPTS', 2)
        timeout = self.get_conf('LIMIT_ORDER_TIMEOUT_SEC', 10)
        force_market = self.get_conf('FORCE_MARKET_ORDER_ON_TIMEOUT', True)
        
        # 체결 데이터 추적용 변수
        executed_ids = []
        total_filled_qty = 0.0
        total_filled_val = 0.0 # qty * price
        
        # 1. MARKET 우선 설정 시
        if priority == 'MARKET':
            try:
                params = {'reduceOnly': True} if reduce_only else {}
                order = await self.exchange.create_market_order(symbol, side, amount, params)
                return order['average'], [order['id']], 'MARKET'
            except Exception as e:
                print(f"❌ Market Order Fail: {e}")
                return None, [], None

        # 2. LIMIT 우선 (지정가 시도 루프)
        remaining_amount = amount
        
        for i in range(attempts):
            try:
                ticker = await self.exchange.fetch_ticker(symbol)
                target_price = ticker['bid'] if side == 'buy' else ticker['ask']
                
                params = {'reduceOnly': True} if reduce_only else {}
                
                print(f"Try Limit {i+1}/{attempts}: {symbol} {side} {remaining_amount} @ {target_price}")
                order = await self.exchange.create_limit_order(symbol, side, remaining_amount, target_price, params)
                current_order_id = order['id']
                
                # 대기
                await asyncio.sleep(timeout)
                
                # 상태 확인
                order_status = await self.exchange.fetch_order(current_order_id, symbol)
                filled = float(order_status['filled'])
                avg_price = float(order_status['average']) if order_status['average'] else target_price
                
                # 체결된 수량이 있다면 기록
                if filled > 0:
                    executed_ids.append(current_order_id)
                    total_filled_qty += filled
                    total_filled_val += (filled * avg_price)
                    remaining_amount -= filled
                
                # 완전 체결 시 종료
                if remaining_amount <= 0:
                    final_wap = total_filled_val / total_filled_qty
                    return final_wap, executed_ids, 'LIMIT'
                
                # 미체결 잔량 존재 -> 주문 취소하고 다음 시도
                # (이미 체결된 부분은 위에서 기록했으므로 취소해도 안전)
                if order_status['status'] == 'open':
                    await self.exchange.cancel_order(current_order_id, symbol)

            except Exception as e:
                print(f"⚠️ Limit Attempt {i+1} Error: {e}")
                await asyncio.sleep(1)

        # 3. Fallback: 시장가 전환 (Dust Check 추가)
        if force_market and remaining_amount > 0:
            try:
                # [Phase 4] 최소 주문 금액 체크 (약 5.5 USDT 안전마진)
                ticker = await self.exchange.fetch_ticker(symbol)
                est_value = remaining_amount * ticker['last']
                
                if est_value < 5.5:
                    print(f"⚠️ 남은 잔량 가치(${est_value:.2f})가 최소 주문 금액 미달로 시장가 전환 포기.")
                    # 지금까지 체결된 것만이라도 반환
                    if total_filled_qty > 0:
                        final_wap = total_filled_val / total_filled_qty
                        return final_wap, executed_ids, 'MIXED'
                    return None, [], None

                print(f"🚀 {symbol} 지정가 실패 -> 남은 {remaining_amount} 시장가 전환")
                params = {'reduceOnly': True} if reduce_only else {}
                order = await self.exchange.create_market_order(symbol, side, remaining_amount, params)
                
                executed_ids.append(order['id'])
                market_filled = float(order['filled'])
                market_price = float(order['average'])
                
                total_filled_qty += market_filled
                total_filled_val += (market_filled * market_price)
                
                final_wap = total_filled_val / total_filled_qty
                return final_wap, executed_ids, 'MIXED' # Limit+Market 섞임
                
            except Exception as e:
                print(f"❌ Market Fallback Fail: {e}")
                
                # 실패했더라도 기존에 일부 지정가 체결된 게 있다면 반환
                if total_filled_qty > 0:
                     final_wap = total_filled_val / total_filled_qty
                     return final_wap, executed_ids, 'LIMIT'
                return None, [], None
        
        # 시장가 전환 옵션이 꺼져있거나 실패했을 때, 체결된 게 있으면 반환
        if total_filled_qty > 0:
            final_wap = total_filled_val / total_filled_qty
            return final_wap, executed_ids, 'LIMIT'
            
        return None, [], None

    async def _fetch_real_commission(self, symbol, order_ids):
        """
        [Phase 4 수정] 단일 ID가 아닌 ID 리스트를 받아 합산 수수료를 계산합니다.
        부분 체결이나 Limit+Market 혼합 체결 시 모든 수수료를 누락 없이 집계합니다.
        """
        if self.mode != 'REAL' or not order_ids: return 0.0
        
        try:
            # 문자열로 변환하여 비교 준비
            target_ids = set(str(oid) for oid in order_ids)
            
            # 최근 거래 내역 조회
            trades = await self.exchange.fetch_my_trades(symbol, limit=50)
            total_fee = 0.0
            
            # BNB 가격 캐싱 (함수 내 1회 조회)
            bnb_price = 0.0
            use_bnb_calc = self.get_conf('USE_BNB_FEE_DISCOUNT', False)
            
            for t in trades:
                # 거래 내역의 orderId가 우리 목록에 있는지 확인
                if str(t['info']['orderId']) in target_ids:
                    if 'fee' in t:
                        cost = float(t['fee']['cost'])
                        currency = t['fee']['currency']
                        
                        if currency == 'USDT':
                            total_fee += cost
                        elif currency == 'BNB':
                            if bnb_price == 0 and use_bnb_calc:
                                try:
                                    ticker = await self.exchange.fetch_ticker('BNB/USDT')
                                    bnb_price = ticker['last']
                                except:
                                    bnb_price = 0
                            
                            if bnb_price > 0:
                                total_fee += (cost * bnb_price)
                                
            return total_fee
        except Exception as e:
            print(f"⚠️ Fee fetch error: {e}")
            return 0.0

    # -----------------------------------------------------------
    # [Data Fetching]
    # -----------------------------------------------------------
    async def fetch_data_and_features(self, symbol):
        """Binance에서 OHLCV 데이터 가져와서 피처 엔지니어링 적용"""
        try:
            # [수정됨] 하드코딩된 '1h' 제거 -> 설정값(TIMEFRAME) 로드
            # 설정이 없으면 기본값 '1h' 사용
            timeframe = self.get_conf('TIMEFRAME', '1h')
            
            # 1. 캔들 데이터 가져오기 (최근 300개)
            ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=300)
            
            # 2. DataFrame 변환
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # 3. 피처 엔지니어링 (중앙 집중식 함수)
            df = apply_features(df)
            
            return df
        except Exception as e:
            # [개선] 단순 print 대신 상세 로그 출력 (어떤 심볼에서 실패했는지 식별)
            print(f"⚠️ Data Fetch Error ({symbol}, TF={self.get_conf('TIMEFRAME', '1h')}): {e}")
            return None

    # -----------------------------------------------------------
    # [Main Logic Loop]
    # -----------------------------------------------------------
    async def run_logic(self):
        """[Phase 4 수정] 메인 매매 로직 (WAP 및 ID 리스트 처리 반영)"""
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
                
                current_price = df['close'].iloc[-1] 
                prev_close = df['close'].iloc[-2]
                prev_rsi = df['RSI_14'].iloc[-2]
                prev_ema = df[f'EMA_{self.EMA_PERIOD}'].iloc[-2]
                prev_atr = df['ATRr_14'].iloc[-2]

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
                        stop_loss = entry_price - (prev_atr * stop_atr) if side == 'buy' else entry_price + (prev_atr * stop_atr)

                    should_close = False
                    reason = ""
                    
                    # 1. 하드 스탑
                    if side == 'buy' and current_price < stop_loss: should_close = True; reason = "StopLoss"
                    elif side == 'sell' and current_price > stop_loss: should_close = True; reason = "StopLoss"
                    
                    # 2. 트레일링 스탑
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

                    # 3. 국면 전환
                    if not should_close:
                        if (entry_regime == 'bull' and regime == r_map['bear']) or \
                           (entry_regime == 'bear' and regime == r_map['bull']):
                             should_close = True; reason = "RegimeChange"

                    if should_close:
                        close_side = 'sell' if side == 'buy' else 'buy'
                        
                        # [Phase 4] 리스트 형태의 order_ids 수신
                        exec_price, order_ids, exec_type = await self.execute_smart_order(symbol, close_side, amount, reduce_only=True)
                        
                        if exec_price:
                            # [Phase 4] 리스트 전달
                            real_fee = await self._fetch_real_commission(symbol, order_ids)
                            
                            duration_min = 0
                            try:
                                entry_time_str = pos_data.get('entry_time')
                                if entry_time_str:
                                    et = datetime.strptime(entry_time_str, '%Y-%m-%d %H:%M:%S')
                                    duration_min = int((datetime.now() - et).total_seconds() / 60)
                            except: pass

                            pnl, _ = await self._save_trade_history(
                                symbol, side, entry_price, exec_price, amount, 
                                real_fee=real_fee, 
                                strategy="HMM_Strategy",
                                entry_regime=entry_regime,
                                exit_reason=reason,
                                duration=duration_min
                            )
                            
                            type_tag = f"[{exec_type}]" if exec_type else ""
                            await self.notification.log(f"💰 [{symbol}] 익절/손절 ({reason}) {type_tag} PnL: ${pnl:.2f} (Duration: {duration_min}m)", level="INFO")
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
                        qty = (allocation * leverage) / current_price
                        
                        # [Phase 4] 리스트 형태의 order_ids 수신
                        exec_price, order_ids, exec_type = await self.execute_smart_order(symbol, signal, qty)
                        
                        if exec_price:
                            stop_atr = self.get_conf('STOP_LOSS_ATR', 2.0)
                            sl_dist = prev_atr * stop_atr
                            sl_price = exec_price - sl_dist if signal == 'buy' else exec_price + sl_dist
                            
                            current_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            
                            await self._upsert_position_to_db(
                                symbol, exec_price, exec_price, exec_price, signal, qty, 
                                target_regime, sl_price, 
                                entry_time=current_time_str
                            )
                            
                            type_tag = f"[{exec_type}]" if exec_type else ""
                            await self.notification.log(f"🚀 [{symbol}] {signal.upper()} {type_tag} 진입 @ {exec_price}", level="INFO")
                            
                            current_pos_count += 1

            return "✅ 매매 로직 실행 완료"

    # -----------------------------------------------------------
    # [Utils] Sync Funding Loop & Backtest Ops
    # -----------------------------------------------------------
    async def sync_funding_fee_loop(self):
        """[PAPER 모드 전용] 8시간마다 실시간 펀딩비 적용 (Long은 지불/수령, Short는 반대)"""
        print("⏳ 펀딩비 동기화 루프 시작...")
        while True:
            try:
                if self.mode == 'PAPER':
                    now = datetime.utcnow()
                    # 바이낸스 펀딩비 정산 시간: 00:00, 08:00, 16:00 (UTC)
                    # 해당 시간대 0~5분 사이에 한 번 실행
                    if now.hour in [0, 8, 16] and now.minute < 5:
                        for sym, pos in list(self.state.items()):
                            try:
                                # 1. 실시간 펀딩비율 조회
                                funding = await self.exchange.fetch_funding_rate(sym)
                                rate = float(funding['fundingRate'])
                                
                                # 2. 포지션 가치 계산
                                position_value = pos['amount'] * pos['entry_price']
                                
                                # 3. 펀딩비 계산 (Cost가 양수면 지불, 음수면 수령)
                                # Long: Rate가 양수면 지불(+), 음수면 수령(-)
                                # Short: Rate가 양수면 수령(-), 음수면 지불(+)
                                funding_cost = position_value * rate
                                
                                real_fee = 0.0
                                if pos['side'] == 'buy':
                                    real_fee = funding_cost
                                else:
                                    real_fee = -funding_cost # Short는 반대
                                
                                if real_fee != 0:
                                    # DB에 펀딩비 기록 (비용 처리)
                                    await self._save_trade_history(sym, 'FUNDING', 0, 0, 0, real_fee=real_fee, trade_type='FUNDING')
                                    print(f"💸 [Funding] {sym} ({pos['side']}) Rate:{rate*100:.4f}% -> Fee: ${real_fee:.4f}")
                                    
                            except Exception as inner_e:
                                print(f"⚠️ {sym} 펀딩비 계산 실패: {inner_e}")
                        
                        # 중복 실행 방지 (1시간 대기)
                        await asyncio.sleep(3600)
                
                # 1분마다 시간 체크
                await asyncio.sleep(60)
            except Exception as e:
                print(f"Funding Sync Error: {e}")
                await asyncio.sleep(60)

    async def get_backtest_result_data(self, record_id):
        """
        백테스트 결과 파일(CSV/ZIP)을 읽어 프론트엔드 그래프용 데이터로 변환합니다.
        다양한 파일 포맷(Daily, Curve, Summary)과 컬럼명(Date, time, Total_Equity, value)을 자동으로 처리합니다.
        """
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute("SELECT csv_path, symbol FROM backtest_history WHERE id=?", (record_id,)) as cursor:
                row = await cursor.fetchone()
        
        if not row: return None
        csv_path, symbol = row
        
        if not csv_path or not os.path.exists(csv_path):
            return None

        try:
            results = []
            
            # [Helper] 데이터프레임 표준화 함수
            def process_df(df, label_name="Equity"):
                # 1. 시간 컬럼 찾기 및 표준화 (time)
                if 'time' in df.columns:
                    pass # 이미 존재
                elif 'Date' in df.columns:
                    df.rename(columns={'Date': 'time'}, inplace=True)
                elif 'timestamp' in df.columns:
                    df.rename(columns={'timestamp': 'time'}, inplace=True)
                else:
                    return None # 시간 컬럼 없음

                # 2. 값 컬럼 찾기 및 표준화 (value)
                val_col = None
                # 우선순위: value -> Total_Equity -> Portfolio_Value -> Equity
                candidates = ['value', 'Total_Equity', 'Portfolio_Value', 'Equity']
                for cand in candidates:
                    if cand in df.columns:
                        val_col = cand
                        break
                
                # 못 찾았다면 마지막 숫자형 컬럼 사용 (휴리스틱)
                if not val_col:
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0: val_col = numeric_cols[-1]
                
                if not val_col: return None

                # 3. 데이터 추출 (JSON 직렬화 가능 형태)
                data = []
                for _, row in df.iterrows():
                    t_val = str(row['time'])
                    # 시간 포맷 정리 (YYYY-MM-DD)
                    if ' ' in t_val: t_val = t_val.split(' ')[0]
                    
                    try:
                        val = float(row[val_col])
                        if not np.isnan(val):
                            data.append({"time": t_val, "value": val})
                    except: pass
                
                if not data: return None
                return {"label": label_name, "data": data}

            # -------------------------------------------------------
            # Case A: ZIP 파일 처리 (리얼 포트폴리오, 배치 테스트)
            # -------------------------------------------------------
            if csv_path.endswith('.zip'):
                with zipfile.ZipFile(csv_path, 'r') as z:
                    namelist = z.namelist()
                    
                    # 1. 메인 그래프 찾기 (Portfolio_Curve.csv 또는 Portfolio_Summary.csv)
                    # Curve: 리얼 포트폴리오용 (time, value)
                    # Summary: 배치용 (Date, Total_Equity)
                    main_file = next((n for n in namelist if n in ['Portfolio_Curve.csv', 'Portfolio_Summary.csv']), None)
                    
                    if main_file:
                        with z.open(main_file) as f:
                            df = pd.read_csv(f)
                            res = process_df(df, "Total Portfolio")
                            if res: results.append(res)
                    
                    # 2. 개별 자산 그래프 찾기 (선택 사항)
                    # 배치 테스트의 경우 개별 코인 데이터(BT_...csv)가 있을 수 있음
                    # 리얼 포트폴리오의 경우 Daily_Stats는 메인과 중복되므로 스킵하거나 추가 가능
                    # 여기서는 'Portfolio_'로 시작하지 않고 'Trades' 로그가 아닌 CSV만 추가
                    for filename in namelist:
                        if filename == main_file: continue
                        if not filename.endswith('.csv'): continue
                        if 'Master_Trade_Log' in filename: continue
                        if 'Daily_Portfolio_Stats' in filename: continue
                        
                        # BT_..._Daily.csv 형태 등
                        if 'BT_' in filename or '_Daily' in filename:
                            # 심볼명 추출 시도
                            label = filename.replace('.csv', '').replace('_Daily', '').split('_')[-1]
                            with z.open(filename) as f:
                                try:
                                    df = pd.read_csv(f)
                                    res = process_df(df, label)
                                    if res: results.append(res)
                                except: pass

            # -------------------------------------------------------
            # Case B: 단일 CSV 파일 처리 (단일 백테스트)
            # -------------------------------------------------------
            else:
                # 단일 파일은 보통 BT_..._Daily.csv 임
                df = pd.read_csv(csv_path)
                res = process_df(df, symbol)
                if res: results.append(res)

            return results if results else None

        except Exception as e:
            print(f"❌ Graph Data Load Error: {e}")
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

    # [NEW] 초기 자본금 조회
    async def get_saved_paper_initial_balance(self):
        val = await self._get_setting('paper_initial_balance')
        return float(val) if val else 10000.0

    async def get_saved_paper_balance(self):
        val = await self._get_setting('paper_balance')
        return float(val) if val else 10000.0

    # [Modified] 초기화 시 초기 자본금도 함께 저장
    async def reset_paper_balance(self, amount):
        await self._set_setting('paper_balance', amount)
        await self._set_setting('paper_initial_balance', amount)
        return amount

    async def set_mode(self, new_mode):
        if new_mode not in ['OFF', 'REAL', 'PAPER']: return "Invalid Mode"
        
        if not self.is_initialized: await self.initialize()
        
        # 모드 변경 저장
        self.mode = new_mode
        await self._set_setting('mode', new_mode)
        
        # [NEW] REAL 모드 진입 시, 현재 실제 잔고를 '초기 자본금'으로 스냅샷 저장
        if new_mode == 'REAL':
            try:
                bal = await self.exchange.fetch_balance()
                current_total = float(bal['total']['USDT'])
                await self._set_setting('real_initial_balance', current_total)
                print(f"🚀 [Real] 실전 매매 시작! 초기 자본금 설정: ${current_total:.2f}")
            except Exception as e:
                print(f"⚠️ [Real] 초기 잔고 조회 실패: {e}")
                # 실패 시 0으로 설정하거나 기존 값 유지
                await self._set_setting('real_initial_balance', 0.0)

        # 상태 및 모델 로드
        self.state = await self._load_positions_from_db()
        if new_mode != 'OFF': 
            await self.load_models()
            
        msg = f"✅ 모드 변경: {new_mode}"
        await self.notification.log(msg, level="SYSTEM")
        return msg

    # [Modified] 증거금 계산 로직 추가
    async def get_balance(self):
        # [NEW] OFF 모드일 때는 잔고를 0으로 리턴하여 UI 초기화
        if self.mode == 'OFF':
            return 0.0, 0.0

        if self.mode == 'PAPER':
            total_bal = await self.get_saved_paper_balance()
            
            # 사용 중인 증거금 계산
            used_margin = 0.0
            leverage = self.get_conf('LEVERAGE', 1.0)
            
            for symbol, pos in self.state.items():
                entry_val = pos['entry_price'] * pos['amount']
                used_margin += entry_val / leverage
                
            free_bal = total_bal - used_margin
            return free_bal, total_bal
            
        # REAL 모드
        try:
            bal = await self.exchange.fetch_balance()
            return bal['free']['USDT'], bal['total']['USDT']
        except: return 0.0, 0.0

    # [NEW] 현재 모드에 맞는 '기준(초기) 자본금' 반환
    async def get_initial_balance(self):
        if self.mode == 'OFF':
            return 0.0 # 정지 상태면 0 리턴 (수익률 0% 표시용)
            
        if self.mode == 'PAPER':
            return await self.get_saved_paper_initial_balance()
            
        if self.mode == 'REAL':
            val = await self._get_setting('real_initial_balance')
            return float(val) if val else 0.0
            
        return 0.0

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
        target_symbols = list(self.state.keys())
        
        for symbol in target_symbols:
            msg = await self.safe_force_close(symbol)
            logs.append(msg)
        
        return "\n".join(logs)

    async def close(self):
        await self.exchange.close()