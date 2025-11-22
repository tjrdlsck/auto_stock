import os
import sys
import asyncio
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import uvicorn

# 프로젝트 모듈 임포트
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import BASE_DIR, MODELS_DIR, DATA_DIR

# 백테스팅 모듈 임포트 (존재한다고 가정)
try:
    import backtest_runner
except ImportError:
    backtest_runner = None

app = FastAPI()

# ---------------------------------------------------------
# [Pydantic Models] 요청 데이터 검증용
# ---------------------------------------------------------
class ControlCommand(BaseModel):
    command: str
    value: Optional[str] = None

class BacktestParams(BaseModel):
    symbol: str = "BTC/USDT"
    initial_cash: float = 10000.0
    train_days: int = 365
    test_days: int = 30
    update_data: bool = False
    is_batch: bool = False

class BalanceReset(BaseModel):
    amount: float

class ConfigUpdateModel(BaseModel):
    settings: Dict[str, Any]

# ---------------------------------------------------------
# [Helper] State Access
# ---------------------------------------------------------
def get_trader():
    if not hasattr(app.state, 'trader') or app.state.trader is None:
        raise HTTPException(status_code=503, detail="Trader not initialized")
    return app.state.trader

def get_hub():
    if not hasattr(app.state, 'hub') or app.state.hub is None:
        raise HTTPException(status_code=503, detail="Notification Hub not initialized")
    return app.state.hub

# ---------------------------------------------------------
# [Routes] 1. Config Management (설정 관리)
# ---------------------------------------------------------
@app.get("/api/config")
async def get_config():
    """현재 설정값 전체 조회"""
    trader = get_trader()
    return trader.config_manager.get_all()

@app.post("/api/config/update")
async def update_config(data: ConfigUpdateModel):
    """설정값 변경 및 저장"""
    trader = get_trader()
    success = trader.config_manager.update_bulk(data.settings)
    if success:
        return {"status": "success", "message": "설정이 업데이트되었습니다.", "config": trader.config_manager.get_all()}
    else:
        raise HTTPException(status_code=500, detail="Failed to save config")

@app.post("/api/config/reset")
async def reset_config():
    """설정값 기본값으로 초기화"""
    trader = get_trader()
    new_conf = trader.config_manager.reset_to_defaults()
    return {"status": "success", "message": "기본값으로 초기화되었습니다.", "config": new_conf}

# ---------------------------------------------------------
# [Routes] 2. System Status & Control (상태 및 제어)
# ---------------------------------------------------------
@app.get("/api/status")
async def get_status():
    """봇의 현재 상태(잔고, 포지션 등) 조회"""
    trader = get_trader()
    
    free, total = await trader.get_balance()
    positions = await trader.get_positions()
    
    # 저장된 모의투자 잔고 가져오기 (설정 화면용)
    saved_paper_bal = trader.get_saved_paper_balance()
    
    # 디스코드 봇 연결 상태 확인
    discord_active = False
    if hasattr(app.state, 'hub') and app.state.hub.discord_bot:
        discord_active = app.state.hub.discord_bot.is_ready()

    return {
        "mode": trader.mode,
        "balance": {"free": free, "total": total},
        "paper_balance": saved_paper_bal,
        "positions": positions,
        "discord_active": discord_active
    }

@app.post("/api/control")
async def control_system(cmd: ControlCommand):
    """봇 제어 (시작, 정지, 모드 변경)"""
    trader = get_trader()
    
    if cmd.command == "set_mode":
        if cmd.value in ['REAL', 'PAPER', 'OFF']:
            msg = await trader.set_mode(cmd.value)
            return {"status": "success", "message": msg}
        else:
            raise HTTPException(status_code=400, detail="Invalid mode")
    
    elif cmd.command == "stop":
        # [수정됨] 정지 시 보유 포지션 전량 청산 로직 실행
        close_log = await trader.close_all_positions()
        
        # 청산 후 모드 OFF 전환
        mode_log = await trader.set_mode('OFF')
        
        combined_msg = f"{close_log}\n{mode_log}"
        return {"status": "success", "message": combined_msg}
        
    return {"status": "error", "message": "Unknown command"}

@app.post("/api/trade/close/{symbol:path}")
async def force_close(symbol: str):
    """특정 포지션 강제 청산 (URL의 symbol 파라미터 처리 개선)"""
    trader = get_trader()
    
    # URL 디코딩 및 포맷 정규화 (예: BTC_USDT -> BTC/USDT)
    # FastAPI는 자동으로 %2F를 /로 디코딩해주지만, 혹시 언더스코어로 오는 경우 대비
    clean_symbol = symbol.replace('_', '/').strip()
    
    msg = await trader.safe_force_close(clean_symbol)
    return {"status": "success", "message": msg}

@app.post("/api/reset_balance")
async def reset_paper_balance(data: BalanceReset):
    """모의투자 잔고 강제 초기화"""
    trader = get_trader()
    new_bal = trader.reset_paper_balance(data.amount)
    return {"status": "success", "balance": new_bal}

# ---------------------------------------------------------
# [Routes] 3. History & Analysis (이력 및 분석)
# ---------------------------------------------------------
@app.get("/api/history")
async def get_history(mode: str = "PAPER"):
    """거래 이력 조회"""
    trader = get_trader()
    data = trader.get_trade_history(target_mode=mode)
    return data

@app.post("/api/backtest")
async def run_backtest(params: BacktestParams):
    """백테스팅 실행 (비동기 처리)"""
    if backtest_runner is None:
        raise HTTPException(status_code=501, detail="Backtest module not found")
        
    hub = get_hub()
    
    # 웹소켓으로 로그를 보내기 위한 콜백 함수
    def log_callback(msg):
        try:
            loop = asyncio.get_running_loop()
            asyncio.run_coroutine_threadsafe(
                hub.log(msg, level="BACKTEST", send_to_discord=False), loop
            )
        except RuntimeError:
            pass

    # blocking 연산이므로 executor에서 실행
    loop = asyncio.get_running_loop()
    
    try:
        if params.is_batch:
            # 전체 포트폴리오 백테스팅
            result = await loop.run_in_executor(
                None,
                lambda: backtest_runner.run_batch_backtest(
                    initial_cash=params.initial_cash,
                    train_days=params.train_days,
                    test_days=params.test_days,
                    update_data=params.update_data,
                    log_func=log_callback
                )
            )
        else:
            # 단일 심볼 백테스팅
            result = await loop.run_in_executor(
                None,
                lambda: backtest_runner.run_walk_forward(
                    symbol=params.symbol,
                    initial_cash=params.initial_cash,
                    train_days=params.train_days,
                    test_days=params.test_days,
                    update_data=params.update_data,
                    log_func=log_callback
                )
            )
            
        if "error" in result:
            raise HTTPException(status_code=500, detail=result['error'])
            
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------------------------------------------
# [Routes] 4. WebSocket & Static Files
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

# 정적 파일 서빙 (index.html)
@app.get("/", response_class=HTMLResponse)
async def read_root():
    # 현재 파일(api_server.py)이 있는 경로의 'web' 폴더 안에서 index.html을 찾음
    base_dir = os.path.dirname(os.path.abspath(__file__))
    html_path = os.path.join(base_dir, "web", "index.html")
    
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    
    # 디버깅을 위해 경로를 출력해주는 것이 좋음
    print(f"❌ 파일을 찾을 수 없음: {html_path}")
    return f"index.html not found at {html_path}"

# ---------------------------------------------------------
# [Server Runner]
# ---------------------------------------------------------
async def run_api_server(app_instance, host="0.0.0.0", port=8000):
    config = uvicorn.Config(app_instance, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)
    print(f"🌍 [API] Server started at http://{host}:{port}")
    await server.serve()