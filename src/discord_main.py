# --- src/discord_main.py 수정본 ---

import matplotlib
matplotlib.use('Agg') 

import os
import discord
import asyncio
import pandas as pd
from discord.ext import commands, tasks
from dotenv import load_dotenv

# 모듈 임포트
from trader import BinanceTrader
import backtest_runner
from data_layer.data_handler import MultiSymbolLoader  # 데이터 수집기
from alpha_layer.brain import Brain                     # AI 학습기

load_dotenv()
TOKEN = os.getenv("DISCORD_BOT_TOKEN")
ADMIN_ID = int(os.getenv("DISCORD_ADMIN_ID"))

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix='!', intents=intents)
trader = BinanceTrader()
is_trading_active = True 

# --- 기존 메신저 브릿지 및 공통 함수 ---
async def send_discord_alert(symbol, side, price, qty, msg_type):
    channel = get_notification_channel()
    if channel:
        color = 0x00ff00 if side.lower() == 'buy' else 0xff0000
        emoji = "🚀" if side.lower() == 'buy' else "📉"
        embed = discord.Embed(title=f"{emoji} BINANCE {side.upper()} {msg_type}", color=color)
        embed.add_field(name="코인", value=f"`{symbol}`", inline=True)
        embed.add_field(name="가격", value=f"`${price:,.4f}`", inline=True)
        embed.add_field(name="수량", value=f"`{qty}`", inline=True)
        embed.set_footer(text="QuantBot V5")
        await channel.send(embed=embed)

def messenger_bridge(symbol, side, price, qty, msg_type):
    if bot.loop.is_running():
        bot.loop.create_task(send_discord_alert(symbol, side, price, qty, msg_type))

trader.messenger = messenger_bridge

def get_notification_channel():
    # 봇이 속한 서버의 첫 번째 채널을 사용 (수정 가능)
    for guild in bot.guilds:
        if guild.text_channels:
            return guild.text_channels[0]
    return None

def is_admin(ctx):
    return ctx.author.id == ADMIN_ID

# ---------------------------------------------------
# [NEW] 스케줄러: 자동 재학습 (24시간 간격)
# ---------------------------------------------------
@tasks.loop(hours=24)
async def auto_retraining_loop():
    """
    1. 최신 데이터 수집
    2. AI 모델 재학습
    """
    channel = get_notification_channel()
    if channel:
        await channel.send("🔄 **[시스템]** 일일 데이터 갱신 및 AI 재학습 시작...")

    print("🔄 [Auto-Retrain] 파이프라인 가동 시작")

    # CPU 집약적 작업이므로 executor에서 실행하여 봇 멈춤 방지
    try:
        loop = asyncio.get_running_loop()
        
        # 1. 데이터 다운로드
        loader = MultiSymbolLoader()
        await loop.run_in_executor(None, loader.run_pipeline)
        
        # 2. 모델 학습
        brain = Brain()
        await loop.run_in_executor(None, brain.run_training)
        
        print("✅ [Auto-Retrain] 학습 완료")
        if channel:
            await channel.send("✅ **[시스템]** AI 모델 업데이트 완료! 최신 데이터가 반영되었습니다.")
            
    except Exception as e:
        print(f"❌ [Auto-Retrain] 실패: {e}")
        if channel:
            await channel.send(f"❌ **[시스템]** 재학습 중 오류 발생: {e}")

@tasks.loop(minutes=60)
async def hourly_trade_loop():
    if is_trading_active:
        print(f"⏰ [스케줄러] 매매 로직 실행...")
        # 봇이 멈추지 않게 비동기로 실행
        msg = await bot.loop.run_in_executor(None, trader.run_logic)
        # (선택) 매 시간 로그를 디스코드에 찍고 싶으면 아래 주석 해제
        # channel = get_notification_channel()
        # if channel: await channel.send(f"`{msg}`")

# ---------------------------------------------------
# 봇 이벤트
# ---------------------------------------------------
@bot.event
async def on_ready():
    print(f'🤖 디스코드 봇 로그인 성공: {bot.user}')
    
    # 트레이딩 루프 시작
    if not hourly_trade_loop.is_running():
        hourly_trade_loop.start()
        
    # 재학습 루프 시작
    if not auto_retraining_loop.is_running():
        auto_retraining_loop.start()

@bot.event
async def on_message(message):
    if message.author.bot:
        return

    print(f"📩 수신된 메시지: '{message.content}'")

    if message.content == "!":
        embed = discord.Embed(title="🤖 퀀트 봇 명령어", description="사용 가능한 명령어 목록입니다.", color=0x3498db)
        embed.add_field(name="!상태", value="현재 잔고 및 포지션 확인", inline=False)
        embed.add_field(name="!백테 [코인] [일수]", value="예: `!백테 BTC/USDT 30`", inline=False)
        embed.add_field(name="!전체백테 [일수]", value="등록된 모든 코인 백테스팅. 예: `!전체백테 180`", inline=False)
        embed.add_field(name="!청산 [코인]", value="예: `!청산 DOGE/USDT`", inline=False)
        embed.add_field(name="!실매매시작 / !모의매매시작 / !정지", value="봇의 매매 모드를 제어합니다.", inline=False)
        await message.channel.send(embed=embed)
        return

    await bot.process_commands(message)

# ---------------------------------------------------
# 명령어 목록
# ---------------------------------------------------
@bot.command(name="백테", aliases=["bt"])
async def cmd_backtest(ctx, symbol: str = "BTC/USDT", days: int = 365):
    if not is_admin(ctx): return
    await ctx.send(f"⏳ **{symbol}** {days}일 백테스팅 계산 중...")
    
    # 파일 경로 반환 안 함 -> summary만 받음
    summary, _ = await bot.loop.run_in_executor(None, lambda: backtest_runner.run_single_backtest(symbol, days))
    await ctx.send(summary)

@bot.command(name="전체백테", aliases=["abt"])
async def cmd_batch_backtest(ctx, days: int = 365):
    if not is_admin(ctx): return
    await ctx.send(f"⏳ 전체 포트폴리오 시뮬레이션 중... ({days}일)")
    
    report = await bot.loop.run_in_executor(None, lambda: backtest_runner.run_batch_backtest(days))
    # 메시지가 길 수 있으므로 나눠서 보내거나 그냥 보냄 (Discord 2000자 제한 주의)
    if len(report) > 1900:
        await ctx.send(report[:1900] + "\n...(생략)")
    else:
        await ctx.send(report)

@bot.command(name="청산", aliases=["close"])
async def cmd_close(ctx, symbol: str):
    if not is_admin(ctx): return
    await ctx.send(f"⚠️ **{symbol}** 강제 청산 시도 중...")
    result = await bot.loop.run_in_executor(None, lambda: trader.force_close(symbol))
    await ctx.send(result)

@bot.command(name="상태", aliases=["status"])
async def status(ctx):
    if not is_admin(ctx): return
    free_usdt, total_usdt = await bot.loop.run_in_executor(None, trader.get_balance)
    positions = await bot.loop.run_in_executor(None, trader.get_positions)
    
    embed = discord.Embed(title="📊 퀀트 봇 상태 리포트", color=0x3498db)
    embed.add_field(name="💰 자산 현황", value=f"총액: **${total_usdt:,.2f}**\n가용: ${free_usdt:,.2f}", inline=False)
    
    if positions:
        pos_str = ""
        for p in positions:
            pnl = p['unrealizedPnl']
            icon = "🟢" if pnl >= 0 else "🔴"
            pos_str += f"**{p['symbol']}** ({p['side']} x{p['leverage']})\n{icon} ${pnl:.2f} (Entry: ${p['entryPrice']:.2f})\n\n"
        embed.add_field(name="📈 포지션", value=pos_str, inline=False)
    else:
        embed.add_field(name="📈 포지션", value="보유 중인 포지션 없음", inline=False)
        
    await ctx.send(embed=embed)

# --- [NEW] 실시간 제어 명령어 ---
@bot.command(name="실매매시작")
async def start_real(ctx):
    if not is_admin(ctx): return
    msg = trader.set_mode('REAL')
    global is_trading_active
    is_trading_active = True # 스케줄러 활성화
    await ctx.send(f"🚨 **주의!** 실제 자산이 투입됩니다.\n{msg}")

@bot.command(name="모의매매시작")
async def start_paper(ctx):
    if not is_admin(ctx): return
    msg = trader.set_mode('PAPER')
    global is_trading_active
    is_trading_active = True # 스케줄러 활성화
    await ctx.send(f"🧪 가상 트레이딩 모드입니다.\n{msg}")

@bot.command(name="정지", aliases=["stop"])
async def stop_bot(ctx):
    if not is_admin(ctx): return
    msg = trader.set_mode('OFF')
    global is_trading_active
    is_trading_active = False # 스케줄러 일시정지
    await ctx.send(f"🛑 모든 매매 활동을 중단합니다.\n{msg}")

bot.run(TOKEN)
