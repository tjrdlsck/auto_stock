import sys
import os
import json
import joblib
import pandas as pd
import pandas_ta as ta
import numpy as np
import ccxt
from datetime import datetime
from dotenv import load_dotenv

# 프로젝트 경로 설정
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import CONFIG, MODELS_DIR

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
        
        # [NEW] 매매 모드 설정 ('OFF', 'REAL', 'PAPER')
        self.mode = 'OFF' 

        # 전략 파라미터 (V5)
        self.RSI_BUY = 55
        self.RSI_SELL = 45
        self.EMA_PERIOD = 20
        self.STOP_LOSS = 0.05
        self.TRAIL_PCT = 0.03
        
        # 상태 관리
        self.STATE_FILE = "trade_state.json"
        self.state = self.load_state()

    def set_mode(self, mode):
        """모드 변경 함수"""
        mode = mode.upper()
        if mode in ['OFF', 'REAL', 'PAPER']:
            self.mode = mode
            return f"✅ 매매 모드가 **{self.mode}** 상태로 변경되었습니다."
        return "❌ 잘못된 모드입니다. (OFF/REAL/PAPER)"

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

    def set_leverage(self, symbol, leverage):
        try:
            self.exchange.set_leverage(leverage, symbol)
        except:
            pass

    def fetch_data_and_features(self, symbol):
        try:
            candles = self.exchange.fetch_ohlcv(symbol, timeframe='1h', limit=100)
            df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            df['Log_Returns'] = np.log(df['close'] / df['close'].shift(1))
            df['Range_Vol'] = (df['high'] - df['low']) / df['close']
            df.ta.rsi(length=14, append=True)
            df.ta.ema(length=self.EMA_PERIOD, append=True)

            window = 30
            features = ['Log_Returns', 'Range_Vol', 'RSI_14']
            for col in features:
                rm = df[col].rolling(window=window).mean()
                rs = df[col].rolling(window=window).std()
                df[f'{col}_Scaled'] = (df[col] - rm) / (rs + 1e-8)
            
            return df.dropna()
        except:
            return None

    def get_hmm_regime(self, df, symbol):
        try:
            clean_symbol = symbol.replace('/', '')
            model_path = os.path.join(MODELS_DIR, f"hmm_{clean_symbol}.pkl")
            
            if not os.path.exists(model_path): return None, None

            model = joblib.load(model_path)
            X = df[['Log_Returns_Scaled', 'Range_Vol_Scaled', 'RSI_14_Scaled']].values
            hidden_states = model.predict(X)
            
            stats = pd.DataFrame(X, columns=['Ret', 'Vol', 'RSI'])
            stats['Regime'] = hidden_states
            regime_means = stats.groupby('Regime')['Ret'].mean()
            
            bear_id = regime_means.idxmin()
            bull_id = regime_means.idxmax()
            
            return hidden_states[-1], {'bull': bull_id, 'bear': bear_id}
        except:
            return None, None

    def execute_order(self, symbol, side, amount, reduce_only=False):
        """모드에 따라 주문 실행 여부 결정"""
        
        # 1. 정지 상태면 즉시 리턴
        if self.mode == 'OFF':
            return None

        # 가격 정보 가져오기 (모의매매에서도 가격은 필요함)
        ticker = self.exchange.fetch_ticker(symbol)
        current_price = ticker['last']
        msg_type = "청산" if reduce_only else "진입"

        # 2. 모의매매 (PAPER)
        if self.mode == 'PAPER':
            log_msg = f"🧪 [모의매매] {symbol} {side.upper()} {msg_type} 시그널! (가격: {current_price}, 수량: {amount})"
            print(log_msg)
            if self.messenger:
                # 모의매매용 별도 알림 함수 호출 or 기존 함수에 플래그 전달
                self.messenger(symbol, side, current_price, amount, f"[모의] {msg_type}")
            return current_price # 가상 체결 가격 반환

        # 3. 실매매 (REAL)
        if self.mode == 'REAL':
            try:
                params = {'reduceOnly': True} if reduce_only else {}
                order = self.exchange.create_market_order(symbol, side, amount, params)
                exec_price = order['average'] if order['average'] else order['price']
                
                print(f"🚀 [실매매] {symbol} {side} 체결 완료")
                if self.messenger:
                    self.messenger(symbol, side, exec_price, amount, msg_type)
                return exec_price
            except Exception as e:
                print(f"❌ 주문 실패: {e}")
                return None

    # ----------------------------------------------
    # [NEW] 디스코드 봇 보고용 함수
    # ----------------------------------------------
    def get_balance(self):
        """현재 USDT 잔고 조회"""
        try:
            balance = self.exchange.fetch_balance()
            return balance['free']['USDT'], balance['total']['USDT']
        except:
            return 0.0, 0.0

    def get_positions(self):
        """현재 활성화된 포지션 조회"""
        active_positions = []
        try:
            # 바이낸스는 모든 심볼 포지션을 줌, 필터링 필요
            positions = self.exchange.fetch_positions()
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
            return []

    def force_close(self, symbol):
        """디스코드 명령어로 특정 심볼 강제 청산"""
        try:
            positions = [p for p in self.exchange.fetch_positions([symbol]) if p['symbol'] == symbol]
            if not positions:
                return f"⚠️ {symbol} 활성화된 포지션이 없습니다."
            
            pos_amt = float(positions[0]['contracts'])
            pos_side = positions[0]['side']
            
            if pos_amt == 0:
                return f"⚠️ {symbol} 포지션 수량이 0입니다."

            # 반대 주문으로 청산
            side = 'sell' if pos_side == 'long' else 'buy'
            price = self.execute_order(symbol, side, pos_amt, reduce_only=True)
            
            if price:
                # 상태 파일에서 삭제
                if symbol in self.state: del self.state[symbol]
                self.save_state()
                return f"✅ **{symbol}** 강제 청산 완료! (가격: {price})"
            else:
                return f"❌ {symbol} 청산 주문 실패."
                
        except Exception as e:
            return f"❌ 강제 청산 중 에러: {e}"


    # ----------------------------------------------
    # 메인 로직 (외부에서 호출)
    # ----------------------------------------------
    def run_logic(self):
        if self.mode == 'OFF':
            return "⏸️ 봇이 정지 상태입니다."
        log_buffer = [] # 로그 저장용
        
        try:
            balance = self.exchange.fetch_balance()
            usdt_balance = balance['free']['USDT']
        except:
            return "❌ 잔고 조회 실패"

        for symbol in CONFIG['SYMBOLS']:
            # 데이터 준비
            df = self.fetch_data_and_features(symbol)
            if df is None: continue
            
            # 현재 값
            curr_price = df['close'].iloc[-1]
            curr_rsi = df['RSI_14'].iloc[-1]
            curr_ema = df[f'EMA_{self.EMA_PERIOD}'].iloc[-1]
            
            # HMM 판단
            regime, r_map = self.get_hmm_regime(df, symbol)
            if regime is None: continue
            
            # 포지션 확인
            positions = [p for p in self.exchange.fetch_positions([symbol]) if p['symbol'] == symbol]
            pos_amt = float(positions[0]['contracts']) if positions else 0.0
            pos_side = positions[0]['side'] if positions else None
            has_position = pos_amt > 0
            
            # 설정 로드
            risk_setting = self.get_risk_setting(symbol)
            leverage = risk_setting['leverage']
            
            # 상태 파일 관리
            if not has_position and symbol in self.state:
                del self.state[symbol]

            # 1. 청산 로직
            if has_position:
                entry_price = float(positions[0]['entryPrice'])
                if symbol not in self.state: self.state[symbol] = {'high': entry_price, 'low': entry_price}
                
                if pos_side == 'long':
                    pnl_pct = (curr_price - entry_price) / entry_price
                    if pnl_pct < -self.STOP_LOSS:
                        self.execute_order(symbol, 'sell', pos_amt, reduce_only=True)
                        continue
                    
                    self.state[symbol]['high'] = max(self.state[symbol]['high'], curr_price)
                    if (self.state[symbol]['high'] - curr_price)/self.state[symbol]['high'] > self.TRAIL_PCT and curr_price > entry_price:
                        self.execute_order(symbol, 'sell', pos_amt, reduce_only=True)
                        continue
                        
                    if regime == r_map['bear']:
                        self.execute_order(symbol, 'sell', pos_amt, reduce_only=True)
                        continue

                elif pos_side == 'short':
                    pnl_pct = (entry_price - curr_price) / entry_price
                    if pnl_pct < -self.STOP_LOSS:
                        self.execute_order(symbol, 'buy', pos_amt, reduce_only=True)
                        continue
                        
                    self.state[symbol]['low'] = min(self.state[symbol]['low'], curr_price)
                    if (curr_price - self.state[symbol]['low'])/self.state[symbol]['low'] > self.TRAIL_PCT and curr_price < entry_price:
                        self.execute_order(symbol, 'buy', pos_amt, reduce_only=True)
                        continue
                        
                    if regime == r_map['bull']:
                        self.execute_order(symbol, 'buy', pos_amt, reduce_only=True)
                        continue

            # 2. 진입 로직
            else:
                self.set_leverage(symbol, leverage)
                invest_amount = usdt_balance * risk_setting['risk_pct'] * leverage
                qty = invest_amount / curr_price
                
                if regime == r_map['bull']:
                    if curr_rsi < self.RSI_BUY and curr_price > curr_ema:
                        self.execute_order(symbol, 'buy', qty)
                
                elif regime == r_map['bear']:
                    if curr_rsi > self.RSI_SELL and curr_price < curr_ema:
                        self.execute_order(symbol, 'sell', qty)

        self.save_state()
        return "✅ 매매 로직 실행 완료"