import os
import sys
import asyncio
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import uvicorn

# 프로젝트 모듈 임포트
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import BASE_DIR, MODELS_DIR, DATA_DIR

# 백테스팅 모듈 임포트
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

class CustomBacktestParams(BacktestParams):
    custom_settings: Dict[str, Any] = {}
    is_real_portfolio: bool = False # [NEW] 리얼 포트폴리오 모드 플래그 추가

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
# [Routes] 1. Config Management
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
# [Routes] 2. System Status & Control
# ---------------------------------------------------------
@app.get("/api/status")
async def get_status():
    """봇의 현재 상태 조회"""
    trader = get_trader()
    
    free, total = await trader.get_balance()
    positions = await trader.get_positions()
    saved_paper_bal = await trader.get_saved_paper_balance()
    
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
        close_log = await trader.close_all_positions()
        mode_log = await trader.set_mode('OFF')
        return {"status": "success", "message": f"{close_log}\n{mode_log}"}
        
    return {"status": "error", "message": "Unknown command"}

@app.post("/api/trade/close/{symbol:path}")
async def force_close(symbol: str):
    """특정 포지션 강제 청산"""
    trader = get_trader()
    clean_symbol = symbol.replace('_', '/').strip()
    msg = await trader.safe_force_close(clean_symbol)
    return {"status": "success", "message": msg}

@app.post("/api/reset_balance")
async def reset_paper_balance(data: BalanceReset):
    """모의투자 잔고 강제 초기화"""
    trader = get_trader()
    new_bal = await trader.reset_paper_balance(data.amount)
    return {"status": "success", "balance": new_bal}

# ---------------------------------------------------------
# [Routes] 3. Backtest Lab
# ---------------------------------------------------------

# 기존 단순 백테스트 (하위 호환용)
@app.post("/api/backtest")
async def run_backtest(params: BacktestParams):
    return await run_custom_backtest(CustomBacktestParams(**params.dict(), custom_settings={}))

# 커스텀 백테스트 실행 및 저장
@app.post("/api/backtest/run_custom")
async def run_custom_backtest(params: CustomBacktestParams):
    if backtest_runner is None:
        raise HTTPException(status_code=501, detail="Backtest module not found")
        
    hub = get_hub()
    trader = get_trader()
    
    # 1. 동적 설정 생성 (기본 설정 + 커스텀 설정 병합)
    current_config = trader.config_manager.get_all().copy()
    if params.custom_settings:
        current_config.update(params.custom_settings)
        
    # 입력받은 학습/테스트 기간을 설정 정보에 포함
    current_config['TRAIN_DAYS'] = params.train_days
    current_config['TEST_DAYS'] = params.test_days
    
    # 로그 콜백
    def log_callback(msg):
        try:
            loop = asyncio.get_running_loop()
            asyncio.run_coroutine_threadsafe(
                hub.log(msg, level="BACKTEST", send_to_discord=False), loop
            )
        except RuntimeError:
            pass

    loop = asyncio.get_running_loop()
    
    try:
        # 2. 실행 분기 (CPU Bound 작업이므로 Executor에서 실행)
        if params.is_real_portfolio:
            # [NEW] 리얼 포트폴리오 모드 실행
            result = await loop.run_in_executor(
                None,
                lambda: backtest_runner.run_real_portfolio_backtest(
                    initial_cash=params.initial_cash,
                    train_days=params.train_days,
                    test_days=params.test_days,
                    update_data=params.update_data,
                    log_func=log_callback,
                    dynamic_config=current_config
                )
            )
            # 결과 저장
            if "error" not in result:
                save_data = {
                    "symbol": "REAL_PF", # 구분자
                    "params": current_config,
                    "roi": result.get('portfolio_roi', 0),
                    "mdd": result.get('avg_mdd', 0),
                    "win_rate": 0, 
                    "trade_count": result.get('trade_count', 0),
                    "final_balance": result.get('final_balance', 0),
                    "csv_path": result.get('csv_path', '')
                }
                await trader.save_backtest_result(save_data)

        elif params.is_batch:
            # 기존 단순 배치 실행
            result = await loop.run_in_executor(
                None,
                lambda: backtest_runner.run_batch_backtest(
                    initial_cash=params.initial_cash,
                    train_days=params.train_days,
                    test_days=params.test_days,
                    update_data=params.update_data,
                    log_func=log_callback,
                    dynamic_config=current_config
                )
            )
            # 배치 결과 저장
            if "error" not in result:
                save_data = {
                    "symbol": "BATCH_SUM",
                    "params": current_config,
                    "roi": result.get('portfolio_roi', 0),
                    "mdd": result.get('avg_mdd', 0),
                    "win_rate": 0, 
                    "trade_count": sum(len(r.get('trades', [])) for r in result.get('details', [])),
                    "final_balance": 0,
                    "csv_path": result.get('csv_path', '')
                }
                await trader.save_backtest_result(save_data)

        else:
            # 단일 코인 실행
            result = await loop.run_in_executor(
                None,
                lambda: backtest_runner.run_walk_forward(
                    symbol=params.symbol,
                    initial_cash=params.initial_cash,
                    train_days=params.train_days,
                    test_days=params.test_days,
                    update_data=params.update_data,
                    log_func=log_callback,
                    dynamic_config=current_config
                )
            )
            # 단일 결과 저장
            if "error" not in result:
                await trader.save_backtest_result(result)
            
        if "error" in result:
            raise HTTPException(status_code=500, detail=result['error'])
            
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/backtest/history")
async def get_backtest_history():
    """백테스트 실행 이력 조회"""
    trader = get_trader()
    return await trader.get_backtest_history()

@app.delete("/api/backtest/history/{record_id}")
async def delete_backtest_history(record_id: int):
    """백테스트 이력 및 파일 삭제"""
    trader = get_trader()
    await trader.delete_backtest_record(record_id)
    return {"status": "success", "message": "Deleted"}

# [NEW] 저장된 백테스트 결과 데이터(차트용) 조회
@app.get("/api/backtest/result/{record_id}")
async def get_backtest_result(record_id: int):
    """특정 백테스트 레코드의 상세 데이터(Equity Curve 등) 반환"""
    trader = get_trader()
    data = await trader.get_backtest_result_data(record_id)
    
    if data is None:
        raise HTTPException(status_code=404, detail="Result data not found or file missing")
        
    return data

@app.get("/api/backtest/download/{record_id}")
async def download_backtest_csv(record_id: int):
    """백테스트 파일(CSV 또는 ZIP) 다운로드"""
    trader = get_trader()
    
    history = await trader.get_backtest_history()
    target = next((item for item in history if item["id"] == record_id), None)
    
    if not target:
        raise HTTPException(status_code=404, detail="Record not found")
        
    file_path = target.get("csv_path")
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    filename = os.path.basename(file_path)
    media_type = 'application/zip' if filename.endswith('.zip') else 'text/csv'
        
    return FileResponse(path=file_path, filename=filename, media_type=media_type)

# ---------------------------------------------------------
# [Routes] 4. WebSocket & Static Files
# ---------------------------------------------------------
@app.get("/api/history")
async def get_history(mode: str = "PAPER"):
    trader = get_trader()
    return await trader.get_trade_history(target_mode=mode)

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

@app.get("/", response_class=HTMLResponse)
async def read_root():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    html_path = os.path.join(base_dir, "web", "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    return f"index.html not found at {html_path}"

# ---------------------------------------------------------
# [Server Runner]
# ---------------------------------------------------------
async def run_api_server(app_instance, host="0.0.0.0", port=8000):
    config = uvicorn.Config(app_instance, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)
    print(f"🌍 [API] Server started at http://{host}:{port}")
    await server.serve()