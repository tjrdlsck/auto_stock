import os

# ---------------------------------------------------------
# [1] 시스템 경로 설정 (기존 유지)
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ---------------------------------------------------------
# [2] 사용자 설정 섹션 (업데이트됨)
# ---------------------------------------------------------
# [수정] CONFIG 딕셔너리 내용을 아래와 같이 업데이트하세요.
CONFIG = {
    # 1. 포트폴리오 대상 코인
    "SYMBOLS": ["BTC/USDT", "ETH/USDT", "SOL/USDT", "DOGE/USDT", "XRP/USDT"],
    
    # 2. 시간봉 설정
    "TIMEFRAME": "1h",
    
    # [수정] 3. 데이터 수집 기간 (약 7년 치 확보)
    "FETCH_DAYS": 2500,
    
    # [NEW] 최소 학습 요구량 (이 기간보다 데이터가 적은 신규 코인은 매매 제외)
    "MIN_TRAIN_DAYS": 1500,
    
    # 4. 레버리지
    "LEVERAGE": 2.0,
    
    # ... (나머지 설정 기존 유지) ...
    "COMMISSION": 0.0005,
    "MAKER_FEE": 0.0002,
    "TAKER_FEE": 0.0005,
    "USE_BNB_FEE_DISCOUNT": True,
    "SLIPPAGE_PCT": 0.0002,
    
    "ORDER_TYPE_PRIORITY": "LIMIT",
    "LIMIT_ORDER_ATTEMPTS": 2,
    "LIMIT_ORDER_TIMEOUT_SEC": 10,
    "FORCE_MARKET_ORDER_ON_TIMEOUT": True,
    
    "MAX_OPEN_POSITIONS": 2,
    
    "RSI_BUY_LOWER": 30,
    "RSI_BUY_UPPER": 65,
    "RSI_SELL_LOWER": 30,
    "RSI_SELL_UPPER": 75,
    
    "STOP_LOSS_ATR": 2.5,
    "TRAIL_TRIGGER_ATR": 2.0,
    "TRAIL_DIST_ATR": 2.0,
    
    "RETRAIN_INTERVAL_HOURS": 24,
    "GENERATE_BACKTEST_IMAGES": True,
    "CANDLE_LIMIT": 300,
}

print(f"✅ 설정 로드 완료: {len(CONFIG['SYMBOLS'])}개 심볼, 수수료 체계: Maker({CONFIG['MAKER_FEE']})/Taker({CONFIG['TAKER_FEE']})")