import os

# ---------------------------------------------------------
# [1] 시스템 경로 설정 (자동 설정)
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# 폴더가 없으면 자동 생성
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ---------------------------------------------------------
# [2] 사용자 설정 섹션 (여기만 수정하세요!)
# ---------------------------------------------------------
CONFIG = {
    # 1. 대상 코인 목록 (거래대금 상위 위주 추천)
    # 비트코인(BTC), 이더리움(ETH), 솔라나(SOL), 리플(XRP), 도지(DOGE)
    "SYMBOLS": ["BTC/USDT", "ETH/USDT", "SOL/USDT", "DOGE/USDT"],
    
    # 2. 캔들 시간봉 (1h, 4h, 1d)
    "TIMEFRAME": "1h",
    
    # 3. 학습용 데이터 수집 기간 (일 단위)
    # 1500일 = 약 4년 (2021년 불장 + 2022년 하락장 + 2023년 회복기 모두 포함)
    "FETCH_DAYS": 1500,
    
    # 4. 백테스팅 설정
    "INITIAL_CASH": 100.0,   # 코인별 할당 자본금 (달러)
    "LEVERAGE": 2.0,           # 레버리지 (1.0 = 현물과 동일)
    
    # 5. 백테스트 기간 (None으로 두면 전체 기간 테스트)
    # 특정 기간만 하려면 '2024-01-01' 형태로 입력 'None'이면 전체 기간
    "BACKTEST_START": "2025-01-01", 
    "BACKTEST_END": "2025-11-10",
    
    # 6. 거래 수수료 (바이낸스 선물 시장가 기준 약 0.04~0.05%)
    "COMMISSION": 0.0005,
}

print(f"✅ 설정 로드 완료: {len(CONFIG['SYMBOLS'])}개 심볼 타겟팅")
