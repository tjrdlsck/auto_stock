import asyncio
import sys
import os
import json # [추가]
import websockets # [추가] 
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

async def websocket_listener(trader, hub):
    """
    [Phase 3 신규] 바이낸스 웹소켓 실시간 리스너
    - 전 종목 시세(!miniTicker@arr)를 구독하여 0.1초 단위 가격 변동 감지
    - 보유 포지션이 있을 경우 trader.check_safety_conditions 호출
    """
    uri = "wss://fstream.binance.com/ws/!miniTicker@arr"
    
    while True:
        try:
            print("🔌 [WebSocket] 바이낸스 퓨처 실시간 시세 서버 연결 시도...")
            async with websockets.connect(uri) as websocket:
                success_msg = "✅ [WebSocket] 연결 성공! 실시간 감시 모드 가동"
                print(success_msg)
                await hub.log(success_msg, level="SYSTEM", send_to_discord=False)
                
                async for message in websocket:
                    # 메인 로직이 정지 상태면 데이터 처리 스킵 (부하 감소)
                    if trader.mode == 'OFF':
                        continue

                    # 보유 포지션이 없으면 계산할 필요 없음
                    if not trader.state:
                        continue

                    data = json.loads(message)
                    # data format: [{'s': 'BTCUSDT', 'c': '50000.00', ...}, ...]
                    
                    for ticker in data:
                        raw_symbol = ticker['s'] # 예: BTCUSDT
                        price = float(ticker['c'])
                        
                        # trader.state의 키는 'BTC/USDT' 형식이므로 변환 필요
                        # 매번 모든 키를 순회하는 것은 비효율적일 수 있으나, 
                        # 포지션 개수가 적으므로(최대 3~5개) 단순 매칭이 가장 안전함
                        target_symbol = None
                        for sym in trader.state.keys():
                            if sym.replace('/', '') == raw_symbol:
                                target_symbol = sym
                                break
                        
                        if target_symbol:
                            # Watchdog: 실시간 가격을 Trader에게 전달 (Locking은 내부 처리)
                            await trader.check_safety_conditions(target_symbol, price)
                            
        except Exception as e:
            err_msg = f"⚠️ [WebSocket] 연결 끊김 ({e}). 5초 후 재접속..."
            print(err_msg)
            # 너무 잦은 로그 전송 방지
            await asyncio.sleep(5)

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
    print("🤖 QUANT BOT HYBRID SYSTEM V2.2 (Real-time Watchdog)")
    print("="*50)

    # 1. 컴포넌트 초기화
    config_manager = ConfigManager()
    hub = NotificationHub()
    trader = BinanceTrader(config_manager, hub)

    # ------------------------------------------------------------------
    # 안전장치: 시작 시 모델 파일 존재 여부 확인 및 자동 생성
    # ------------------------------------------------------------------
    existing_models = glob.glob(os.path.join(MODELS_DIR, "*.pkl"))
    
    if not existing_models:
        print("\n⚠️ [Init] 학습된 모델 파일이 없습니다. (첫 실행으로 간주)")
        print("🚀 [Init] 초기 AI 모델 학습을 시작합니다...")
        
        loop = asyncio.get_running_loop()
        try:
            loader = MultiSymbolLoader()
            await loop.run_in_executor(None, loader.run_pipeline)
            
            brain = Brain()
            await loop.run_in_executor(None, brain.run_training)
            print("✅ [Init] 초기 학습 완료!\n")
        except Exception as e:
            print(f"❌ [Init] 초기 학습 중 오류: {e}")
            return
    else:
        print(f"✅ [Init] 기존 모델 파일 감지됨 ({len(existing_models)}개).")
    # ------------------------------------------------------------------

    # Trader 초기화
    await trader.initialize()

    # 의존성 주입
    app.state.trader = trader
    app.state.hub = hub
    
    try:
        # [수정] 스케줄러 및 리스너 등록
        # 1. 정각 매매 스케줄러 (Brain)
        scheduler_task = asyncio.create_task(trading_scheduler(trader, hub))
        
        # 2. 주기적 재학습 스케줄러
        retrain_task = asyncio.create_task(periodic_retraining(trader, hub))
        
        # 3. [NEW] 실시간 웹소켓 리스너 (Reflex)
        websocket_task = asyncio.create_task(websocket_listener(trader, hub))

        # 통합 실행
        await asyncio.gather(
            run_api_server(app, host="0.0.0.0", port=New_Port),
            start_discord_bot(trader, hub),
            scheduler_task,
            retrain_task,
            websocket_task # 추가됨
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