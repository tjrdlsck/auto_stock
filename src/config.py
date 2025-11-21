import os

# ---------------------------------------------------------
# [1] 시스템 경로 설정
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# 데이터 및 모델 저장 디렉토리 자동 생성
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ---------------------------------------------------------
# [2] 사용자 설정 섹션
# ---------------------------------------------------------
CONFIG = {
    # 1. 포트폴리오 대상 코인
    # 다양한 섹터로 분산하는 것이 좋습니다.
    "SYMBOLS": ["BTC/USDT", "ETH/USDT", "SOL/USDT", "DOGE/USDT", "XRP/USDT"],
    
    # 2. 시간봉 설정 (1시간봉 기준 전략)
    "TIMEFRAME": "1h",
    
    # 3. 데이터 수집 기간 (일)
    # HMM 학습을 위해 충분한 과거 데이터가 필요합니다.
    "FETCH_DAYS": 1500,
    
    # 4. 기본 레버리지
    # 백테스팅 및 실매매 공통 적용
    "LEVERAGE": 2.0,
    
    # 5. 거래 비용 설정 (바이낸스 선물 기준 근사치)
    "COMMISSION": 0.0005,  # 수수료 0.05%
    "SLIPPAGE_PCT": 0.0002, # 슬리피지 0.02% 가정
    
    # -----------------------------------------------------
    # [NEW] 6. 자금 관리 (Portfolio Management)
    # -----------------------------------------------------
    # 동시에 보유할 수 있는 최대 코인 개수입니다.
    # 예: 3으로 설정 시, 전체 자금을 3등분하여 진입하며 4번째 코인은 진입하지 않습니다.
    "MAX_OPEN_POSITIONS": 5,
    
    # -----------------------------------------------------
    # [NEW] 7. 전략 파라미터 (V13 - Unconstrained Trend)
    # -----------------------------------------------------
    # 7-1. 진입 조건 (RSI 필터 완화)
    # Bull 진입: Price > EMA & RSI가 [30, 65] 사이
    "RSI_BUY_LOWER": 30,
    "RSI_BUY_UPPER": 65,
    
    # Bear 진입: Price < EMA & RSI가 [35, 70] 사이
    "RSI_SELL_LOWER": 35,
    "RSI_SELL_UPPER": 70,
    
    # 7-2. 청산 및 관리 조건 (ATR 기반 동적 대응)
    "STOP_LOSS_ATR": 2.0,      # 진입 시 초기 손절 거리 (ATR x 2)
    "TRAIL_TRIGGER_ATR": 2.0,  # 수익이 ATR x 2 이상 발생 시 트레일링 시작
    "TRAIL_DIST_ATR": 2.0,     # 고점(Long)/저점(Short)에서 ATR x 2 간격 유지
    
    # -----------------------------------------------------
    # [NEW] 8. 자동화 설정
    # -----------------------------------------------------
    "RETRAIN_INTERVAL_HOURS": 24, # 24시간마다 AI 모델 재학습 수행
    
    # 9. 시스템 실행 설정
    "GENERATE_BACKTEST_IMAGES": True,
    "LIMIT_ORDER_TIMEOUT_SEC": 10,      # 지정가 주문 대기 시간
    "FORCE_MARKET_ORDER_ON_TIMEOUT": True, # 시간 초과 시 시장가 전환 여부
    "CANDLE_LIMIT": 300,                # 지표 계산을 위해 가져올 최소 캔들 수
}

print(f"✅ 설정 로드 완료: {len(CONFIG['SYMBOLS'])}개 심볼 타겟팅, 포트폴리오 슬롯: {CONFIG['MAX_OPEN_POSITIONS']}개")