import asyncio
import sys
import os
from datetime import datetime, timedelta

# 현재 디렉토리를 모듈 검색 경로에 추가 (import 오류 방지)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 모듈 임포트
from config_manager import ConfigManager
from notification import NotificationHub
from trader import BinanceTrader
from api_server import app, run_api_server
from discord_main import start_discord_bot

New_Port = 58000  # API 서버 포트 설정

async def trading_scheduler(trader, hub):
    """
    [핵심 스케줄러]
    1. 봇 시작 직후: 다음 정각(XX:00:00)까지 대기합니다.
    2. 정각 도달 시: 매매 로직을 실행하고, 다시 다음 정각까지 대기합니다.
    """
    # 1. 첫 실행을 위한 대기 시간 계산
    now = datetime.now()
    # 현재 시간에서 1시간 뒤의 시각을 구한 뒤, 분/초를 0으로 맞춤 (다음 정각)
    next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
    wait_seconds = (next_hour - now).total_seconds()
    
    start_msg = f"⏳ [Scheduler] 매매 시스템 대기 중... 첫 실행: {next_hour.strftime('%H:%M:%S')} ({int(wait_seconds)}초 후)"
    print(start_msg)
    await hub.log(start_msg, level="SYSTEM", send_to_discord=False)
    
    # 다음 정각까지 Sleep
    await asyncio.sleep(wait_seconds)

    # 2. 정각 도달 후 무한 루프 (1시간 간격)
    while True:
        try:
            current_time_str = datetime.now().strftime('%H:%M')
            print(f"\n⏰ [Scheduler] 정각 도달! 매매 로직 실행 ({current_time_str})")
            
            # (1) 매매 로직 실행
            result = await trader.run_logic()
            # 결과 로그 전송 (디스코드 포함)
            await hub.log(result, level="SYSTEM", send_to_discord=True)
            
            # (2) 상태 업데이트 (웹 UI 갱신용)
            free, total = await trader.get_balance()
            positions = await trader.get_positions()
            status_data = {
                "type": "status_update",
                "balance": {"free": free, "total": total},
                "positions": positions,
                "mode": trader.mode
            }
            await hub.broadcast_status(status_data)

        except Exception as e:
            err_msg = f"❌ 매매 스케줄러 오류 발생: {e}"
            print(err_msg)
            await hub.log(err_msg, level="ERROR", send_to_discord=True)
        
        # 3. 다음 1시간 대기 (시간 밀림 방지를 위해 매번 재계산)
        now = datetime.now()
        next_run = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
        sleep_time = (next_run - now).total_seconds()
        
        # 음수가 나오면(연산이 1시간 넘게 걸림) 즉시 실행하도록 0 처리
        if sleep_time < 0: sleep_time = 0
        
        await asyncio.sleep(sleep_time)

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

    # ---------------------------------------------------------------------------
    # [FIX] 추가됨: 스케줄러가 대기하더라도 DB가 먼저 생성되도록 명시적 초기화 호출
    # ---------------------------------------------------------------------------
    print("🔧 [Init] DB 테이블 생성 및 초기화 수행...")
    await trader.initialize()

    # -----------------------------------------------------
    # 2. 의존성 주입 (Dependency Injection)
    # -----------------------------------------------------
    # FastAPI 서버에 Trader와 Hub 주입
    app.state.trader = trader
    app.state.hub = hub
    
    # -----------------------------------------------------
    # 3. 통합 실행 (Concurrency)
    # -----------------------------------------------------
    
    try:
        # [수정됨] 스케줄러를 별도 태스크로 생성 (디스코드 의존성 제거)
        scheduler_task = asyncio.create_task(trading_scheduler(trader, hub))

        # 세 개의 비동기 루프(API서버, 디스코드봇, 매매스케줄러)를 병렬로 실행
        await asyncio.gather(
            run_api_server(app, host="0.0.0.0", port=New_Port),
            start_discord_bot(trader, hub),
            scheduler_task
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