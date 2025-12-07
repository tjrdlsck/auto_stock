import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

class Config:
    """
    프로젝트 설정 클래스 (Gemini API & Backtest Ver.)
    """

    # ---------------------------------------------------------
    # 1. Exchange & Account Settings
    # ---------------------------------------------------------
    EXCHANGE_NAME = "binance"
    
    # 데이터 수집용 키 (없으면 공개 데이터만 수집 가능하나, 제한이 있을 수 있음)
    BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
    BINANCE_SECRET_KEY = os.getenv("BINANCE_SECRET_KEY")
    
    # [NEW] Google Gemini API Settings
    # .env 파일에 GOOGLE_API_KEY를 설정해야 합니다.
    GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
    GEMINI_MODEL = "gemma-3-4b-it"  # 빠르고 저렴한 모델 선택

    # INFO: 매매 내역 화면 출력 (평소 디버깅용)
    # WARNING: 화면 조용히, 에러만 출력 (최적화용)
    LOG_LEVEL = "INFO"

    # 2. Target Market (핵심 수정: 일봉 전략이므로 1d 권장)
    SYMBOL = "BTC/USDT"
    TIMEFRAME = "1d"  # [수정] 1h -> 1d (변동성 돌파의 정석)
    
    TOTAL_DAYS_TO_FETCH = 1825  # 1년치 데이터
    TEST_DAYS = 90         # 최근 6개월 백테스트
    
    # 3. Strategy Parameters (VBO + Dynamic K)
    # 변동성 돌파 전략 전용 설정
    VBO_K_WINDOW = 20      # Noise Ratio 평균 기간
    SMA_PERIOD = 50        # 추세 판단용 이동평균
    ATR_PERIOD = 14        # 변동성 지표 기간
    
    # 4. Risk Management
    RISK_PER_TRADE = 0.20      # 자산의 2% 리스크
    LEVERAGE = 5.0            # 레버리지 1배
    STOP_LOSS_MULTIPLIER = 2.0 # ATR * 2.0 손절

    # 백테스트 비용 설정
    INITIAL_BALANCE = 100.0
    TRADING_FEE_RATE = 0.0005  # 0.05%
    SLIPPAGE = 0.0003          # 0.02%

    # ---------------------------------------------------------
    # [NEW] Trailing Stop Settings
    # ---------------------------------------------------------
    USE_TRAILING_STOP = True       # True: 사용, False: 미사용 (일반 VBO로 동작)
    TRAILING_STOP_MULTIPLIER = 2.0 # 예: ATR의 3배 간격으로 따라감

    # System
    SAVE_DIR = "data"

    @classmethod
    def validate(cls):
        """설정 검증"""
        print(f"[Config] {cls.SYMBOL} ({cls.TIMEFRAME}) | Strategy: Volatility Breakout")
        if not cls.GEMINI_API_KEY:
            raise ValueError("⚠️ Google Gemini API Key is missing. Please set GOOGLE_API_KEY in .env")
        print(f"[Config] Loaded. Model: {cls.GEMINI_MODEL}, Symbol: {cls.SYMBOL}")
        print(f"[Config] Data Strategy: Fetch {cls.TOTAL_DAYS_TO_FETCH} days -> Test last {cls.TEST_DAYS} days.")

if __name__ == "__main__":
    try:
        Config.validate()
    except Exception as e:
        print(e)