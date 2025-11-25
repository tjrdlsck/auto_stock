import matplotlib
matplotlib.use('Agg') 

import os
import discord
import asyncio
import requests
import pandas as pd
import io
from discord.ext import commands, tasks
from dotenv import load_dotenv

from config import CONFIG, DATA_DIR
import backtest_runner
from data_layer.data_handler import MultiSymbolLoader
from alpha_layer.brain import Brain

load_dotenv()

# ---------------------------------------------------------
# [Safe Env Loading] 환경변수 안전 로드
# ---------------------------------------------------------
TOKEN = os.getenv("DISCORD_BOT_TOKEN", "")
admin_env = os.getenv("DISCORD_ADMIN_ID", "")

if admin_env.isdigit():
    ADMIN_ID = int(admin_env)
else:
    ADMIN_ID = 0

DISCORD_ENABLED = bool(TOKEN and ADMIN_ID)
HEARTBEAT_URL = "https://hc-ping.com/800eca18-a428-4a27-87ee-071a92eb0298"

# ---------------------------------------------------------
# [Bot Class]
# ---------------------------------------------------------
class QuantBot(commands.Bot):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trader = None  
        self.hub = None     

    async def send_log(self, message: str, level: str):
        if not self.is_ready(): return

        channel = None
        for guild in self.guilds:
            if guild.text_channels:
                channel = guild.text_channels[0]
                break
        
        if channel:
            prefix = "ℹ️"
            if "ERROR" in level: prefix = "🚨"
            elif "WARN" in level: prefix = "⚠️"
            elif "REAL" in level: prefix = "🚀"
            elif "PAPER" in level: prefix = "🧪"
            elif "SYSTEM" in level: prefix = "🔧"
            elif "BACKTEST" in level: prefix = "📊"
            
            content = f"{prefix} **[{level}]** {message}"
            if "긴급" in message or "ERROR" in level:
                content += "\n<@everyone>"
            
            try:
                await channel.send(content)
            except: pass

intents = discord.Intents.default()
intents.message_content = True
bot = QuantBot(command_prefix='!', intents=intents)

def is_admin(ctx):
    return ctx.author.id == ADMIN_ID

# ---------------------------------------------------------
# [Loops]
# ---------------------------------------------------------
@tasks.loop(minutes=1)
async def heartbeat_loop():
    try: 
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, requests.get, HEARTBEAT_URL)
    except: pass

@tasks.loop(hours=1)
async def hourly_trade_loop():
    if not bot.trader: return
    print(f"\n⏰ [Hourly Loop] 매매 로직 시작")
    try:
        # [Modified] await 추가
        result = await bot.trader.run_logic()
        if bot.hub:
            await bot.hub.log(result, level="SYSTEM", send_to_discord=True)
            
        # [Modified] await 추가
        free, total = await bot.trader.get_balance()
        positions = await bot.trader.get_positions()
        
        status_data = {
            "type": "status_update",
            "balance": {"free": free, "total": total},
            "positions": positions,
            "mode": bot.trader.mode
        }
        await bot.hub.broadcast_status(status_data)
            
    except Exception as e:
        if bot.hub:
            await bot.hub.log(f"매매 루프 에러: {e}", level="ERROR")

@tasks.loop(hours=24)
async def auto_retraining_loop():
    if not bot.trader: return
    await bot.hub.log("🔄 [시스템] AI 모델 재학습 시작", level="SYSTEM")
    try:
        loop = asyncio.get_running_loop()
        
        loader = MultiSymbolLoader()
        # 데이터 수집은 동기 함수(API 요청 많음) -> Executor 실행 권장
        await loop.run_in_executor(None, loader.run_pipeline)
        
        brain = Brain()
        # 모델 학습은 CPU 부하 높음 -> Executor 실행 권장
        await loop.run_in_executor(None, brain.run_training)
        
        # 학습 완료 후 Trader의 모델 리로드
        await bot.trader.load_models()
        
        await bot.hub.log("✅ [시스템] AI 모델 재학습 및 리로드 완료", level="SYSTEM")
    except Exception as e:
        await bot.hub.log(f"⚠️ [오류] 재학습 실패: {e}", level="ERROR")

@auto_retraining_loop.before_loop
async def before_retraining():
    if DISCORD_ENABLED:
        await bot.wait_until_ready()

# ---------------------------------------------------------
# [Events & Commands]
# ---------------------------------------------------------
@bot.event
async def on_ready():
    print(f'🤖 디스코드 봇 로그인: {bot.user}')
    if not hourly_trade_loop.is_running(): hourly_trade_loop.start()
    if not auto_retraining_loop.is_running(): auto_retraining_loop.start()
    if not heartbeat_loop.is_running(): heartbeat_loop.start()
    
    # [NEW] 펀딩비 동기화 루프 실행 (비동기 태스크로 등록)
    if bot.trader:
        asyncio.create_task(bot.trader.sync_funding_fee_loop())

@bot.command(name="상태", aliases=["status"])
async def status(ctx):
    if not is_admin(ctx): return
    if not bot.trader: return
    
    # [Modified] await 추가
    free, total = await bot.trader.get_balance()
    positions = await bot.trader.get_positions()
    
    mode = bot.trader.mode
    embed = discord.Embed(title=f"📊 상태 ({mode})", color=0x3498db)
    embed.add_field(name="💰 자산", value=f"${total:,.2f} (가용: ${free:,.2f})", inline=False)
    
    if positions:
        pos_str = ""
        for p in positions:
            pnl = p.get('unrealizedPnl', 0.0)
            icon = "🟢" if pnl >= 0 else "🔴"
            pos_str += f"**{p['symbol']}** ({p['side']} x{p['leverage']})\n{icon} ${pnl:.2f} (Entry: ${p['entryPrice']:.4f})\n\n"
        embed.add_field(name=f"📈 포지션 ({len(positions)})", value=pos_str, inline=False)
    else:
        embed.add_field(name="포지션", value="없음", inline=False)
        
    await ctx.send(embed=embed)

@bot.command(name="실매매시작")
async def start_real(ctx):
    if not is_admin(ctx): return
    # [Modified] await 추가
    msg = await bot.trader.set_mode('REAL')
    await ctx.send(msg)

@bot.command(name="모의매매시작")
async def start_paper(ctx):
    if not is_admin(ctx): return
    # [Modified] await 추가
    msg = await bot.trader.set_mode('PAPER')
    await ctx.send(msg)

@bot.command(name="정지", aliases=["stop"])
async def stop_bot(ctx):
    if not is_admin(ctx): return
    # [Modified] await 추가
    msg = await bot.trader.set_mode('OFF')
    await ctx.send(msg)

@bot.command(name="청산", aliases=["close"])
async def cmd_close(ctx, symbol: str):
    if not is_admin(ctx): return
    # [Modified] await 추가
    result = await bot.trader.safe_force_close(symbol)
    await ctx.send(result)

# ---------------------------------------------------------
# [Backtest Commands]
# ---------------------------------------------------------
is_backtesting = False

@bot.command(name="백테", aliases=["bt"])
async def cmd_backtest(ctx, symbol: str = "BTC/USDT", test_days: int = 30, train_days: int = 365, initial_cash: float = 10000.0):
    """
    단일 코인 백테스팅 (데이터 업데이트 포함)
    """
    global is_backtesting
    if not is_admin(ctx): return
    if is_backtesting: return await ctx.send("⏳ 다른 작업이 진행 중입니다.")

    is_backtesting = True
    await ctx.send(f"📥 **{symbol}** 데이터 업데이트 및 백테스팅 시작...\n(검증: {test_days}일, 학습: {train_days}일, 시드: ${initial_cash:,.0f})")

    def log_callback(msg):
        if bot.hub:
            asyncio.run_coroutine_threadsafe(
                bot.hub.log(msg, level="BACKTEST", send_to_discord=False), 
                bot.loop
            )

    try:
        loop = asyncio.get_running_loop()
        # 백테스트는 CPU 연산이 많으므로 Executor에서 실행
        result = await loop.run_in_executor(
            None, 
            lambda: backtest_runner.run_walk_forward(
                symbol, initial_cash, train_days, test_days, update_data=True, log_func=log_callback
            )
        )

        if "error" in result:
            await ctx.send(f"❌ 백테스팅 실패: {result['error']}")
        else:
            summary = (
                f"📊 **{symbol} 백테스트 결과**\n"
                f"━━━━━━━━━━━━━━━━\n"
                f"💰 최종 잔고: **${result['final_balance']:,.2f}**\n"
                f"📈 수익률 (ROI): **{result['roi']:+.2f}%**\n"
                f"📉 총 거래 횟수: {result['trade_count']}회"
            )
            
            df = pd.DataFrame(result['trades'])
            if not df.empty:
                csv_buffer = io.StringIO()
                df.to_csv(csv_buffer, index=False)
                csv_buffer.seek(0)
                file = discord.File(fp=io.BytesIO(csv_buffer.getvalue().encode()), filename=f"BT_{symbol.replace('/','')}.csv")
                await ctx.send(content=summary, file=file)
            else:
                await ctx.send(f"{summary}\n(거래 내역 없음)")

    except Exception as e:
        await ctx.send(f"❌ 에러 발생: {e}")
    finally:
        is_backtesting = False

@bot.command(name="전체백테", aliases=["abt"])
async def cmd_batch_backtest(ctx, test_days: int = 30, train_days: int = 365, initial_cash: float = 10000.0):
    """
    전체 포트폴리오 백테스팅 (데이터 업데이트 포함)
    """
    global is_backtesting
    if not is_admin(ctx): return
    if is_backtesting: return await ctx.send("⏳ 다른 작업이 진행 중입니다.")

    is_backtesting = True
    await ctx.send(f"🚀 **전체 포트폴리오** 시뮬레이션 시작 (시간이 소요됩니다)...")

    def log_callback(msg):
        if bot.hub:
            asyncio.run_coroutine_threadsafe(
                bot.hub.log(msg, level="BACKTEST", send_to_discord=False), 
                bot.loop
            )

    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, 
            lambda: backtest_runner.run_batch_backtest(
                initial_cash, train_days, test_days, update_data=True, log_func=log_callback
            )
        )

        if "error" in result:
            await ctx.send(f"❌ 실행 실패: {result['error']}")
        else:
            msg = [f"🏆 **포트폴리오 종합 결과** (ROI: {result['portfolio_roi']:+.2f}%)", "━━━━━━━━━━━━━━━━"]
            for res in result['details']:
                icon = "🟢" if res['roi'] >= 0 else "🔴"
                msg.append(f"{icon} **{res['symbol']}**: {res['roi']:+.2f}% (${res['final_balance']:,.0f})")
            
            msg.append(f"\n🔎 평균 수익률: **{result['avg_roi']:+.2f}%**")
            
            full_text = "\n".join(msg)
            if len(full_text) > 1900:
                await ctx.send(f"🏆 **종합 ROI**: {result['portfolio_roi']:+.2f}%\n🔎 **평균 ROI**: {result['avg_roi']:+.2f}%")
            else:
                await ctx.send(full_text)

    except Exception as e:
        await ctx.send(f"❌ 에러 발생: {e}")
    finally:
        is_backtesting = False

# ---------------------------------------------------------
# [Entry Point]
# ---------------------------------------------------------
async def start_discord_bot(shared_trader, shared_hub):
    bot.trader = shared_trader
    bot.hub = shared_hub
    
    if not DISCORD_ENABLED:
        msg = "⚠️ .env에 디스코드 설정이 없어 봇 기능을 비활성화합니다."
        print(msg)
        await shared_hub.log(msg, level="WARN", send_to_discord=False)
        
        # 봇 로그인 없이 루프만 수동 실행
        if not hourly_trade_loop.is_running(): hourly_trade_loop.start()
        if not auto_retraining_loop.is_running(): auto_retraining_loop.start()
        
        # [Modified] 펀딩비 루프 실행 (비동기)
        if shared_trader:
            asyncio.create_task(shared_trader.sync_funding_fee_loop())
        
        return

    shared_hub.register_discord_bot(bot)
    print("🚀 디스코드 봇 시작 중...")
    try:
        await bot.start(TOKEN)
    except Exception as e:
        print(f"❌ 디스코드 로그인 실패: {e}")
        await shared_hub.log(f"디스코드 로그인 실패: {e}", level="ERROR", send_to_discord=False)