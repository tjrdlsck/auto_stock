import asyncio
import os
import backtest_runner
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional
import uvicorn

# ---------------------------------------------------------
# [Models] 요청 데이터 검증용 Pydantic 모델
# ---------------------------------------------------------
class ControlCommand(BaseModel):
    command: str
    value: str

class ConfigUpdate(BaseModel):
    settings: Dict[str, Any]

class BacktestRequest(BaseModel):
    symbol: str = "BTC/USDT"
    initial_cash: float = 10000.0
    train_days: int = 365     # 학습 데이터 기간
    test_days: int = 30       # 검증(테스트) 기간
    update_data: bool = False # 데이터 최신화 여부
    is_batch: bool = False    # 전체 포트폴리오 테스트 여부

# [NEW] 잔고 초기화 요청 모델
class ResetBalanceRequest(BaseModel):
    amount: float = 10000.0

# ---------------------------------------------------------
# [FastAPI App Setup]
# ---------------------------------------------------------
app = FastAPI(title="QuantBot API", version="1.0.0")

# CORS 설정
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 프론트엔드 파일 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(BASE_DIR, 'web')

# ---------------------------------------------------------
# [Helpers] 의존성 객체 가져오기
# ---------------------------------------------------------
def get_trader():
    if not hasattr(app.state, "trader") or app.state.trader is None:
        raise HTTPException(status_code=503, detail="Trader not initialized")
    return app.state.trader

def get_hub():
    if not hasattr(app.state, "hub") or app.state.hub is None:
        raise HTTPException(status_code=503, detail="Notification Hub not initialized")
    return app.state.hub

# ---------------------------------------------------------
# [Endpoints] Frontend Serving (CDN 방식)
# ---------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def read_root():
    """루트 접속 시 index.html 반환"""
    index_file = os.path.join(WEB_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return "<h1>Frontend file not found. Please create src/web/index.html</h1>"

# ---------------------------------------------------------
# [Endpoints] REST API
# ---------------------------------------------------------

@app.get("/api/status")
async def get_status():
    """봇의 현재 상태 (모드, 자산, 포지션, 디스코드 연결 여부) 반환"""
    trader = get_trader()
    hub = get_hub()
    
    free, total = await trader.get_balance()
    positions = await trader.get_positions()
    
    # Hub에 discord_bot 객체가 등록되어 있으면 활성화된 것으로 간주
    discord_status = (hub.discord_bot is not None)
    
    # [수정] 모드와 상관없이 저장된 모의투자 잔고를 조회하여 응답에 포함
    saved_paper_bal = trader.get_saved_paper_balance()
    
    return {
        "mode": trader.mode,
        "balance": {
            "free": free,
            "total": total
        },
        "paper_balance": saved_paper_bal, # [추가됨]
        "positions": positions,
        "is_running": True,
        "discord_active": discord_status
    }

@app.post("/api/control")
async def control_bot(cmd: ControlCommand):
    trader = get_trader()
    if cmd.command == "set_mode":
        msg = await trader.set_mode(cmd.value)
        return {"status": "success", "message": msg, "mode": trader.mode}
    elif cmd.command == "stop":
        msg = await trader.set_mode("OFF")
        return {"status": "success", "message": msg}
    raise HTTPException(status_code=400, detail="Unknown command")

@app.post("/api/trade/close/{symbol}")
async def force_close(symbol: str):
    # URL 인코딩된 슬래시 처리
    if '-' in symbol: symbol = symbol.replace('-', '/')
    trader = get_trader()
    msg = await trader.safe_force_close(symbol)
    return {"status": "success", "message": msg}

@app.get("/api/config")
async def get_config():
    trader = get_trader()
    return trader.config_manager.get_all()

@app.post("/api/config")
async def update_config(data: ConfigUpdate):
    trader = get_trader()
    trader.config_manager.update_bulk(data.settings)
    await get_hub().log(f"⚙️ [Web] 설정 변경됨: {list(data.settings.keys())}", level="SYSTEM")
    return {"status": "success", "message": "Configuration updated"}

@app.get("/api/history")
async def get_trade_history(mode: Optional[str] = None):
    """거래 이력 조회 (차트 데이터 포함)"""
    trader = get_trader()
    # mode가 'undefined'나 빈 문자열로 올 경우 None으로 처리
    target_mode = mode if mode and mode not in ['undefined', 'null'] else None
    return trader.get_trade_history(target_mode=target_mode)

# [NEW] 모의투자 잔고 초기화 엔드포인트
@app.post("/api/reset_balance")
async def reset_balance(req: ResetBalanceRequest):
    """모의투자 잔고를 특정 금액으로 초기화"""
    trader = get_trader()
    new_balance = trader.reset_paper_balance(req.amount)
    
    msg = f"💵 [시스템] 모의투자 잔고가 ${new_balance:,.0f}로 초기화되었습니다."
    await get_hub().log(msg, level="SYSTEM")
    
    return {"status": "success", "new_balance": new_balance}

# ---------------------------------------------------------
# [New Endpoint] Non-blocking Backtest with Live Logs
# ---------------------------------------------------------
@app.post("/api/backtest")
async def run_backtest_endpoint(req: BacktestRequest):
    hub = get_hub()
    loop = asyncio.get_running_loop()
    
    def log_callback(msg):
        asyncio.run_coroutine_threadsafe(
            hub.log(msg, level="BACKTEST", send_to_discord=False),
            loop
        )

    try:
        if req.is_batch:
            result = await loop.run_in_executor(
                None, 
                lambda: backtest_runner.run_batch_backtest(
                    initial_cash=req.initial_cash,
                    train_days=req.train_days,
                    test_days=req.test_days,
                    update_data=req.update_data,
                    log_func=log_callback
                )
            )
        else:
            result = await loop.run_in_executor(
                None, 
                lambda: backtest_runner.run_walk_forward(
                    symbol=req.symbol, 
                    initial_cash=req.initial_cash,
                    train_days=req.train_days,
                    test_days=req.test_days,
                    update_data=req.update_data,
                    log_func=log_callback
                )
            )
        
        if "error" in result:
             raise HTTPException(status_code=400, detail=result["error"])
             
        return result

    except Exception as e:
        print(f"Backtest Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
# ---------------------------------------------------------
# [WebSockets]
# ---------------------------------------------------------
@app.websocket("/ws/logs")
async def websocket_endpoint(websocket: WebSocket):
    hub = get_hub()
    await hub.connect_websocket(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        hub.disconnect_websocket(websocket)
    except Exception:
        hub.disconnect_websocket(websocket)

# ---------------------------------------------------------
# [Runner]
# ---------------------------------------------------------
async def run_api_server(app_instance, host="0.0.0.0", port=8000):
    config = uvicorn.Config(app_instance, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)
    print(f"🌐 API 서버 시작: http://{host}:{port}")
    await server.serve()