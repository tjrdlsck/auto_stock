# ---------------------------------------------------
# src/discord_main.py (최종 수정본)
# ---------------------------------------------------
import matplotlib
matplotlib.use('Agg') # [맥북 필수] GUI 없이 백그라운드 실행

import os
import discord
import asyncio
import pandas as pd
from discord.ext import commands, tasks
from dotenv import load_dotenv
from trader import BinanceTrader
import backtest_runner 

# 설정 로드
load_dotenv()
TOKEN = os.getenv("DISCORD_BOT_TOKEN")
ADMIN_ID = int(os.getenv("DISCORD_ADMIN_ID"))

# [중요] 인텐트 설정 (메시지 읽기 권한)
intents = discord.Intents.default()
intents.message_content = True # 이게 True여야 !를 읽습니다.

bot = commands.Bot(command_prefix='!', intents=intents)

trader = BinanceTrader()
is_trading_active = True 

# --- (중간 생략: 비동기 브릿지, trader 연결 등 기존과 동일) ---
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
    for guild in bot.guilds:
        if guild.text_channels: return guild.text_channels[0]
    return None

def is_admin(ctx):
    return ctx.author.id == ADMIN_ID

# ---------------------------------------------------
# 봇 이벤트
# ---------------------------------------------------
@bot.event
async def on_ready():
    print(f'🤖 디스코드 봇 로그인 성공: {bot.user}')
    if not hourly_trade_loop.is_running():
        hourly_trade_loop.start()

# [핵심] 메시지 수신 이벤트
@bot.event
async def on_message(message):
    # 봇 자신의 메시지는 무시
    if message.author.bot:
        return

    # [디버깅] 봇이 무슨 메시지를 듣는지 터미널에 출력
    # 만약 여기가 빈칸으로 나오면 포털 설정 문제입니다.
    print(f"📩 수신된 메시지: '{message.content}'")

    # ! 만 쳤을 때 도움말
    if message.content == "!":
        embed = discord.Embed(title="🤖 퀀트 봇 명령어", description="사용 가능한 명령어 목록입니다.", color=0x3498db)
        embed.add_field(name="!상태", value="현재 잔고 및 포지션 확인", inline=False)
        embed.add_field(name="!백테 [코인] [일수]", value="예: `!백테 BTC/USDT 30`", inline=False)
        embed.add_field(name="!청산 [코인]", value="예: `!청산 DOGE/USDT`", inline=False)
        embed.add_field(name="!정지 / !시작", value="매매 로직 제어", inline=False)
        await message.channel.send(embed=embed)
        return

    # 이 줄이 있어야 다른 명령어(!상태 등)가 작동함
    await bot.process_commands(message)

@tasks.loop(minutes=60)
async def hourly_trade_loop():
    if is_trading_active:
        print(f"⏰ [스케줄러] 매매 로직 실행...")
        await bot.loop.run_in_executor(None, trader.run_logic)

# ---------------------------------------------------
# 명령어 목록
# ---------------------------------------------------
@bot.command(name="백테", aliases=["backtest", "bt"])
async def cmd_backtest(ctx, symbol: str = "BTC/USDT", days: int = 365):
    if not is_admin(ctx): return
    await ctx.send(f"⏳ **{symbol}** 최근 {days}일 데이터로 백테스팅 중...")
    loop = asyncio.get_running_loop()
    try:
        summary, img_path, excel_path = await loop.run_in_executor(None, lambda: backtest_runner.run_single_backtest(symbol, days))
        files = []
        if img_path and os.path.exists(img_path): files.append(discord.File(img_path, filename="chart.png"))
        if excel_path and os.path.exists(excel_path): files.append(discord.File(excel_path, filename="report.xlsx"))
        await ctx.send(content=summary, files=files)
        if img_path and os.path.exists(img_path): os.remove(img_path)
        if excel_path and os.path.exists(excel_path): os.remove(excel_path)
    except Exception as e:
        await ctx.send(f"❌오류: {e}")

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

@bot.command(name="정지")
async def stop_bot(ctx):
    if not is_admin(ctx): return
    global is_trading_active
    is_trading_active = False
    await ctx.send("🔴 매매 루프 정지")

@bot.command(name="시작")
async def start_bot(ctx):
    if not is_admin(ctx): return
    global is_trading_active
    is_trading_active = True
    await ctx.send("✅ 매매 루프 재시작")

bot.run(TOKEN)