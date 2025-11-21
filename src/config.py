import os

# ---------------------------------------------------------
# [1] 시스템 경로 설정
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ---------------------------------------------------------
# [2] 사용자 설정 섹션
# ---------------------------------------------------------
CONFIG = {
    # 1. 대상 코인
    "SYMBOLS": ["BTC/USDT", "ETH/USDT", "SOL/USDT", "DOGE/USDT"],
    
    # 2. 시간봉
    "TIMEFRAME": "1h",
    
    # 3. 데이터 수집 기간 (일)
    "FETCH_DAYS": 1500,
    
    # 4. 기본 레버리지 (백테스팅/실매매 공통)
    "LEVERAGE": 2.0,
    
    # 5. 수수료 및 슬리피지
    "COMMISSION": 0.0005,
    "SLIPPAGE_PCT": 0.0002,
    
    # 6. 전략 파라미터 (V10.5)
    "ADX_THRESHOLD": 25,       # 추세/횡보 구분 기준
    "BB_BANDWIDTH_LIMIT": 0.015, # 볼린저밴드 최소 폭 (1.5%)
    
    # 7. 시스템 설정
    "GENERATE_BACKTEST_IMAGES": True,
    "LIMIT_ORDER_TIMEOUT_SEC": 10,
    "FORCE_MARKET_ORDER_ON_TIMEOUT": True,
    "CANDLE_LIMIT": 300, # 지표 계산을 위한 최소 데이터
}

print(f"✅ 설정 로드 완료: {len(CONFIG['SYMBOLS'])}개 심볼 타겟팅")