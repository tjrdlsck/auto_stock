import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

class Config:
    """
    프로젝트 설정 클래스 (Step 1: Portfolio Parameter Expansion)
    - 종목별 맞춤 파라미터(SL_MULT, VBO_K 등)를 지원하도록 확장
    """
    
    # ---------------------------------------------------------
    # 1. Target Market & Timeframe
    # ---------------------------------------------------------
    EXCHANGE_NAME = "binance"
    SYMBOL = "BTC/USDT" # 기본 심볼 (main.py에서 덮어씌워짐)
    TIMEFRAME = "1h"    # 1시간봉 (핵심)
    
    # 데이터 수집: 1시간봉이므로 넉넉하게 5년치
    TOTAL_DAYS_TO_FETCH = 365 * 5
    TEST_DAYS = 1400

    # ---------------------------------------------------------
    # 2. Portfolio Settings (핵심 수정)
    # 종목별 성격에 맞춰 '손절폭(SL)'과 '추세(SMA)'를 다르게 설정함
    # ---------------------------------------------------------
    PORTFOLIO = {
        "BTC/USDT": {
            "SMA_PERIOD": 50,       # 일봉 기준 50일선
            "VBO_K_WINDOW": 20,
            "ALLOCATION": 0.3,      # 비중 40%
            "LEVERAGE": 3.0,        # 레버리지 3배 (안전하게 하향 조정 권장)
            "SL_MULT": 2.5,
            "TRAILING_MULT": 4.0,   # ATR 4배 (Wide SL)
            "RISK_PER_TRADE": 0.20  # 트레이드 당 리스크 20% (격리 모드 감안)
        },
        "ETH/USDT": {
            "SMA_PERIOD": 30,       # 일봉 기준 30일선
            "VBO_K_WINDOW": 20,
            "ALLOCATION": 0.4,      # 비중
            "LEVERAGE": 3.0,      
            "SL_MULT": 4.0,  
            "TRAILING_MULT": 2.0,   # ATR 4배 (Wide SL)
            "RISK_PER_TRADE": 0.30
        },
        "SOL/USDT": {
            "SMA_PERIOD": 30,       # 일봉 기준 30일선
            "VBO_K_WINDOW": 20,
            "ALLOCATION": 0.3,      # 비중
            "LEVERAGE": 3.0,      
            "SL_MULT": 4.0,  
            "TRAILING_MULT": 2.0,   # ATR 4배 (Wide SL)
            "RISK_PER_TRADE": 0.30
        }
    }

    # ---------------------------------------------------------
    # 3. Default Strategy Parameters (Fallback)
    # 포트폴리오 설정이 없을 때 사용하는 기본값들
    # ---------------------------------------------------------
    VBO_K_WINDOW = 20      
    SMA_PERIOD = 30        
    ATR_PERIOD = 14        
    
    RISK_PER_TRADE = 0.20      
    LEVERAGE = 3.0             
    STOP_LOSS_MULTIPLIER = 3.0 
    
    USE_TRAILING_STOP = True       
    TRAILING_STOP_MULTIPLIER = 3.0 

    # ---------------------------------------------------------
    # 4. System & Fees
    # ---------------------------------------------------------
    BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
    BINANCE_SECRET_KEY = os.getenv("BINANCE_SECRET_KEY")
    
    LOG_LEVEL = "INFO" # 백테스트 시엔 INFO, 최적화 시엔 WARNING

    # [신규] 펀딩비 계산용 (8시간마다 발생한다고 가정)
    # 상승장 평균 0.01%, 과열 시 0.03% 등 다양하지만 보수적으로 0.01% 적용
    FUNDING_RATE_8H = 0.0001 # 0.01%

    # 백테스트 기본 자금
    INITIAL_BALANCE = 100.0
    TRADING_FEE_RATE = 0.0005  # 0.05%
    SLIPPAGE = 0.0003          # 0.03%

    # System
    SAVE_DIR = "data"

    @classmethod
    def validate(cls):
        print(f"[Config] Loaded. Portfolio: {list(cls.PORTFOLIO.keys())}")
        print(f"[Config] Timeframe: {cls.TIMEFRAME} (Intra-day Logic Ready)")

if __name__ == "__main__":
    Config.validate()