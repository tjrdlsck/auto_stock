import asyncio
from typing import List, Optional
from datetime import datetime

# FastAPI의 WebSocket 타입을 힌팅용으로 가져옵니다.
try:
    from fastapi import WebSocket
except ImportError:
    # 아직 설치 전이라면 Any로 대체 (실행 시에는 설치 필요)
    from typing import Any as WebSocket

class NotificationHub:
    def __init__(self):
        # 웹소켓으로 연결된 클라이언트(브라우저) 목록
        self.active_connections: List[WebSocket] = []
        
        # 디스코드 봇 인스턴스를 저장할 변수 (나중에 main.py에서 주입)
        self.discord_bot = None

    def register_discord_bot(self, bot_instance):
        """
        메인 실행 시 디스코드 봇 객체를 등록하는 메서드
        """
        self.discord_bot = bot_instance
        print("✅ [Hub] 디스코드 봇이 알림 허브에 연결되었습니다.")

    async def connect_websocket(self, websocket: WebSocket):
        """
        웹 클라이언트 접속 처리
        """
        await websocket.accept()
        self.active_connections.append(websocket)
        # 접속 환영 메시지 전송
        await websocket.send_text(f"[{datetime.now().strftime('%H:%M:%S')}] [SYSTEM] 실시간 로그 서버에 연결되었습니다.")

    def disconnect_websocket(self, websocket: WebSocket):
        """
        웹 클라이언트 접속 해제 처리
        """
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def log(self, message: str, level: str = "INFO", send_to_discord: bool = True):
        """
        [핵심] 로그를 받아 웹소켓과 디스코드 양쪽으로 방송(Broadcast)
        """
        # 시간 포맷팅
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # 1. 웹 UI용 포맷 (터미널 느낌)
        # 예: [12:00:00] [INFO] 매수 주문 체결 완료
        formatted_msg = f"[{timestamp}] [{level}] {message}"

        # 2. 웹소켓 전송 (비동기 병렬 처리 고려 가능하나, 여기선 순차 처리)
        # 연결이 끊긴 소켓은 리스트에서 제거
        disconnected_clients = []
        for connection in self.active_connections:
            try:
                await connection.send_text(formatted_msg)
            except Exception:
                disconnected_clients.append(connection)
        
        # 죽은 연결 정리
        for dead_client in disconnected_clients:
            self.disconnect_websocket(dead_client)

        # 3. 디스코드 전송 (옵션)
        # INFO 레벨 이상이거나 중요 메시지일 경우만 보낼 수도 있음
        if send_to_discord and self.discord_bot:
            try:
                # Discord Bot에 구현될 send_log 메서드 호출 (추후 구현 예정)
                # await 없이 task로 던질 수도 있음
                await self.discord_bot.send_log(message, level)
            except Exception as e:
                print(f"⚠️ [Hub] 디스코드 전송 실패: {e}")

        # 4. 서버 콘솔에도 출력
        print(formatted_msg)

    async def broadcast_status(self, status_data: dict):
        """
        봇의 상태(잔고, 포지션 등)가 변경되었을 때 웹소켓에 JSON 데이터 전송
        (React 프론트엔드 상태 업데이트용)
        """
        for connection in self.active_connections:
            try:
                # 텍스트 대신 JSON 전송
                await connection.send_json(status_data)
            except Exception:
                pass