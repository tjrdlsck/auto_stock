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
CONFIG = {
    # 1. 포트폴리오 대상 코인
    "SYMBOLS": ["BTC/USDT", "ETH/USDT", "SOL/USDT", "DOGE/USDT", "XRP/USDT"],
    
    # 2. 시간봉 설정
    "TIMEFRAME": "1h",
    
    # 3. 데이터 수집 기간
    "FETCH_DAYS": 1500,
    
    # 4. 레버리지
    "LEVERAGE": 2.0,
    
    # -----------------------------------------------------
    # [5. 수수료 및 슬리피지 설정 (고도화됨)]
    # -----------------------------------------------------
    # 구버전 호환용 (백테스트에서 Taker 기준 보수적 적용 시 사용)
    "COMMISSION": 0.0005,
    
    # [NEW] 상세 수수료 설정 (Binance VIP 0 기준)
    "MAKER_FEE": 0.0002,  # 0.02% (지정가 체결 시)
    "TAKER_FEE": 0.0005,  # 0.05% (시장가 체결 시)
    
    # [NEW] BNB 수수료 할인 사용 여부 (실매매 PnL 보정용)
    # True일 경우 실매매 로직에서 BNB 가격을 조회하여 수수료를 환산 계산함
    "USE_BNB_FEE_DISCOUNT": True,
    
    # 백테스팅용 슬리피지 가정치
    "SLIPPAGE_PCT": 0.0002, 
    
    # -----------------------------------------------------
    # [6. 스마트 주문 실행 설정 (Smart Execution)]
    # -----------------------------------------------------
    # 주문 우선순위: "LIMIT" (지정가 시도 후 시장가) 또는 "MARKET" (즉시 시장가)
    "ORDER_TYPE_PRIORITY": "LIMIT",
    
    # 지정가 주문 시도 횟수 (예: 2회 시도 후 미체결 시 시장가 전환)
    "LIMIT_ORDER_ATTEMPTS": 2,
    
    # 지정가 주문 대기 시간 (초 단위)
    "LIMIT_ORDER_TIMEOUT_SEC": 10,
    
    # 타임아웃/실패 시 시장가로 강제 집행 여부
    "FORCE_MARKET_ORDER_ON_TIMEOUT": True,
    
    # -----------------------------------------------------
    # [7. 자금 관리 (Portfolio Management)]
    # -----------------------------------------------------
    "MAX_OPEN_POSITIONS": 2,
    
    # -----------------------------------------------------
    # [8. 전략 파라미터 (HMM Strategy)]
    # -----------------------------------------------------
    # 진입 조건
    "RSI_BUY_LOWER": 30,
    "RSI_BUY_UPPER": 65,
    "RSI_SELL_LOWER": 30,
    "RSI_SELL_UPPER": 75,
    
    # 청산 및 관리 조건 (ATR 기반)
    "STOP_LOSS_ATR": 2.5,
    "TRAIL_TRIGGER_ATR": 2.0,
    "TRAIL_DIST_ATR": 2.0,
    
    # -----------------------------------------------------
    # [9. 시스템 설정]
    # -----------------------------------------------------
    "RETRAIN_INTERVAL_HOURS": 24,
    "GENERATE_BACKTEST_IMAGES": True,
    "CANDLE_LIMIT": 300,
}

print(f"✅ 설정 로드 완료: {len(CONFIG['SYMBOLS'])}개 심볼, 수수료 체계: Maker({CONFIG['MAKER_FEE']})/Taker({CONFIG['TAKER_FEE']})")