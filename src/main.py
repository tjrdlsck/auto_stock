import asyncio
import sys
import os

# 현재 디렉토리를 모듈 검색 경로에 추가 (import 오류 방지)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 모듈 임포트
from config_manager import ConfigManager
from notification import NotificationHub
from trader import BinanceTrader
from api_server import app, run_api_server
from discord_main import start_discord_bot

New_Port = 48000  # API 서버 포트 설정

async def main():
    print("="*50)
    print("🤖 QUANT BOT HYBRID SYSTEM V2 Starting...")
    print("="*50)

    # -----------------------------------------------------
    # 1. 핵심 컴포넌트 초기화 (Singleton Pattern)
    # -----------------------------------------------------
    print("🔧 [Init] 설정 관리자 및 알림 허브 로드...")
    config_manager = ConfigManager()
    hub = NotificationHub()
    
    print("🔧 [Init] Binance Trader 인스턴스 생성...")
    # Trader에게 Config와 Hub를 주입하여 생성
    trader = BinanceTrader(config_manager, hub)

    # -----------------------------------------------------
    # 2. 의존성 주입 (Dependency Injection)
    # -----------------------------------------------------
    # FastAPI 서버에 Trader와 Hub 주입
    app.state.trader = trader
    app.state.hub = hub
    
    # (디스코드 봇은 start_discord_bot 함수 호출 시 인자로 전달됨)

    # -----------------------------------------------------
    # 3. 통합 실행 (Concurrency)
    # -----------------------------------------------------
    
    try:
        # 두 개의 비동기 루프(서버, 봇)를 병렬로 실행
        await asyncio.gather(
            run_api_server(app, host="0.0.0.0", port=New_Port),
            start_discord_bot(trader, hub)
        )
    except asyncio.CancelledError:
        print("\n🛑 시스템 종료 요청 감지.")
    except Exception as e:
        print(f"\n❌ 치명적 오류 발생: {e}")
    finally:
        # [수정됨] Trader 리소스 정리 호출
        if 'trader' in locals():
            await trader.close()
            
        print("👋 시스템이 종료되었습니다.")

if __name__ == "__main__":
    try:
        if sys.platform == 'win32':
            # 윈도우 환경에서 Event Loop 정책 설정 (SelectorEventLoop 권장)
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
            
        asyncio.run(main())
    except KeyboardInterrupt:
        # Ctrl+C 종료 시 깔끔하게 처리
        pass