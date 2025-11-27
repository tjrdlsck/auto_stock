import asyncio
import sys
import os
from datetime import datetime, timedelta

# 현재 디렉토리를 모듈 검색 경로에 추가 (import 오류 방지)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 모듈 임포트
from config import MODELS_DIR
from config_manager import ConfigManager
from notification import NotificationHub
from trader import BinanceTrader
from api_server import app, run_api_server
from discord_main import start_discord_bot
from alpha_layer.brain import Brain
from data_layer.data_handler import MultiSymbolLoader
import glob 

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



async def periodic_retraining(trader, hub, interval_hours=24):
    """
    [독립 스케줄러] 주기적으로 AI 모델을 재학습시키고 Trader에 리로드 신호를 보냅니다.
    """
    print(f"🔄 [Retrain] 재학습 스케줄러 대기 모드 진입 ({interval_hours}시간 주기)")
    
    while True:
        try:
            # 설정된 주기만큼 대기 (초 단위 환산)
            # 봇 시작 직후에는 모델이 로드되어 있으므로, 대기 후 실행합니다.
            await asyncio.sleep(interval_hours * 3600)
            
            msg = "🔄 [시스템] 정기 AI 모델 재학습 프로세스 시작..."
            print(msg)
            await hub.log(msg, level="SYSTEM", send_to_discord=True)
            
            loop = asyncio.get_running_loop()
            
            # 1. 데이터 수집 (CPU Bound -> Executor)
            loader = MultiSymbolLoader()
            # 데이터 수집은 시간이 오래 걸리므로 Executor 사용
            await loop.run_in_executor(None, loader.run_pipeline)
            
            # 2. 모델 학습 (CPU Bound -> Executor)
            brain = Brain()
            await loop.run_in_executor(None, brain.run_training)
            
            # 3. Trader에게 최신 모델 리로드 요청
            if trader:
                await trader.load_models()
                
            success_msg = "✅ [시스템] 재학습 완료 및 최신 모델(기준표 포함) 리로드 성공"
            print(success_msg)
            await hub.log(success_msg, level="SYSTEM", send_to_discord=True)
            
        except Exception as e:
            err_msg = f"⚠️ [시스템] 재학습 프로세스 실패: {e}"
            print(err_msg)
            await hub.log(err_msg, level="ERROR", send_to_discord=True)
            # 에러 발생 시 10분 대기 후 재시도 방지 (무한 루프 방어)
            await asyncio.sleep(600)

# [수정 위치: main() 함수 전체 교체 또는 내용 수정]

async def main():
    print("="*50)
    print("🤖 QUANT BOT HYBRID SYSTEM V2.1 Starting...")
    print("="*50)

    # 1. 컴포넌트 초기화
    config_manager = ConfigManager()
    hub = NotificationHub()
    trader = BinanceTrader(config_manager, hub)

    # ------------------------------------------------------------------
    # [NEW] 안전장치: 시작 시 모델 파일 존재 여부 확인 및 자동 생성
    # ------------------------------------------------------------------
    # models 폴더 내에 .pkl 파일이 하나라도 있는지 확인
    existing_models = glob.glob(os.path.join(MODELS_DIR, "*.pkl"))
    
    if not existing_models:
        print("\n⚠️ [Init] 학습된 모델 파일이 없습니다. (첫 실행으로 간주)")
        print("🚀 [Init] 초기 AI 모델 학습을 시작합니다. (약 1~2분 소요될 수 있음)...")
        
        loop = asyncio.get_running_loop()
        try:
            # 데이터 수집부터 학습까지 한 번 수행
            loader = MultiSymbolLoader()
            await loop.run_in_executor(None, loader.run_pipeline)
            
            brain = Brain()
            await loop.run_in_executor(None, brain.run_training)
            print("✅ [Init] 초기 학습 완료! 모델과 기준표(_meta.json)가 생성되었습니다.\n")
        except Exception as e:
            print(f"❌ [Init] 초기 학습 중 치명적 오류 발생: {e}")
            # 학습 실패 시 봇 종료 (불완전한 상태로 시작하는 것보다 안전)
            return
    else:
        print(f"✅ [Init] 기존 모델 파일 감지됨 ({len(existing_models)}개). 학습 과정을 건너뜁니다.")
    # ------------------------------------------------------------------

    # Trader 초기화 (이제 모델 파일이 반드시 존재하므로 안전하게 로드됨)
    await trader.initialize()

    # 의존성 주입
    app.state.trader = trader
    app.state.hub = hub
    
    try:
        # 스케줄러들 등록
        scheduler_task = asyncio.create_task(trading_scheduler(trader, hub))
        retrain_task = asyncio.create_task(periodic_retraining(trader, hub))

        # 통합 실행
        await asyncio.gather(
            run_api_server(app, host="0.0.0.0", port=New_Port),
            start_discord_bot(trader, hub),
            scheduler_task,
            retrain_task
        )
    except asyncio.CancelledError:
        print("\n🛑 시스템 종료 요청 감지.")
    except Exception as e:
        print(f"\n❌ 치명적 오류 발생: {e}")
    finally:
        if 'trader' in locals():
            await trader.close()
        print("👋 시스템이 종료되었습니다.")

if __name__ == "__main__":
    try:
        if sys.platform == 'win32':
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        asyncio.run(main())
    except KeyboardInterrupt:
        pass