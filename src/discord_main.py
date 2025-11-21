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
from config import DATA_DIR

load_dotenv()
TOKEN = os.getenv("DISCORD_BOT_TOKEN")
ADMIN_ID = int(os.getenv("DISCORD_ADMIN_ID"))

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix='!', intents=intents)
trader = BinanceTrader()

# --- 기존 메신저 브릿지 ---                                  
async def send_discord_alert(symbol, side, price, qty, msg_type, image_path=None, mention_everyone=False):
    channel = get_notification_channel()
    if channel:
        color = 0x00ff00 if side.lower() == 'buy' else 0xff0000
        emoji = "🚀" if side.lower() == 'buy' else "📉"
        embed = discord.Embed(title=f"{emoji} BINANCE {side.upper()} {msg_type}", color=color)
        embed.add_field(name="코인", value=f"`{symbol}`", inline=True)
        embed.add_field(name="가격", value=f"`${price:,.4f}`", inline=True)
        embed.add_field(name="수량", value=f"`{qty}`", inline=True)
        embed.set_footer(text="QuantBot V7.5")

        if image_path and os.path.exists(image_path):
            file = discord.File(image_path, filename="backtest_result.png")
            embed.set_image(url="attachment://backtest_result.png")
            await channel.send(embed=embed, file=file, content="@everyone" if mention_everyone else None)
        else:
            await channel.send(embed=embed, content="@everyone" if mention_everyone else None)

def messenger_bridge(symbol, side, price, qty, msg_type, image_path=None, mention_everyone=False):
    if bot.loop.is_running():
        bot.loop.create_task(send_discord_alert(symbol, side, price, qty, msg_type, image_path, mention_everyone))

trader.messenger = messenger_bridge

def get_notification_channel():
    for guild in bot.guilds:
        if guild.text_channels:
            return guild.text_channels[0]
    return None

def is_admin(ctx):
    return ctx.author.id == ADMIN_ID

# --- 스케줄러 및 이벤트 핸들러 생략 (기존과 동일) ---        
# (auto_retraining_loop, hourly_trade_loop, on_ready 등은 그대로 유지) 
@tasks.loop(hours=24)
async def auto_retraining_loop():
    # ... (기존 코드 동일)
    pass

@tasks.loop(minutes=60)
async def hourly_trade_loop():
    # ... (기존 코드 동일)
    pass

@bot.event
async def on_ready():
    print(f'🤖 디스코드 봇 로그인 성공: {bot.user}')
    if not hourly_trade_loop.is_running(): hourly_trade_loop.start()
    if not auto_retraining_loop.is_running(): auto_retraining_loop.start()

@bot.event
async def on_message(message):
    if message.author.bot: return

    if message.content == "!":
        embed = discord.Embed(title="🤖 퀀트 봇 명령어 가이드", description="입력 순서를 정확히 지켜주세요.", color=0x3498db)

        embed.add_field(
            name="📊 단일 코인 백테스팅",
            value="**`!백테 [코인] [검증일] [학습일]`**\n예시: `!백테 BTC/USDT 30 365`\n👉 최근 30일 검증 (직전 365일 학습)",
            inline=False
        )
        embed.add_field(
            name="📚 전체 포트폴리오 백테스팅",
            value="**`!전체백테 [검증일] [학습일]`**\n예시: `!전체백테 30 365`",
            inline=False
        )
        embed.add_field(name="기타", value="`!상태`, `!청산 [코인]`, `!실매매시작`, `!정지`", inline=False)
        embed.set_footer(text="Tip: WFA 방식이라 시간이 조금 걸립니다.")
        await message.channel.send(embed=embed)
        return

    await bot.process_commands(message)

# --- 명령어 섹션 ---                                         

is_backtesting = False

@bot.command(name="백테", aliases=["bt"])
async def cmd_backtest(ctx, symbol: str = "BTC/USDT", test_days: int = 30, train_days: int = 365):
    global is_backtesting
    if not is_admin(ctx): return

    if is_backtesting:
        await ctx.send("⏳ **이미 다른 백테스팅이 실행 중입니다.**")
        return

    is_backtesting = True
    try:
        # step_days는 test_days에 따라 자동 조정              
        step_days = 30 if test_days >= 30 else test_days

        await ctx.send(f"⏳ **{symbol}** WFA 백테스트 시작\n" 
                       f"🎯 검증 기간: 최근 **{test_days}일**\n" 
                       f"📚 학습 기간: 직전 **{train_days}일** (Rolling)\n" 
                       f"📉 ADX 필터: **{backtest_runner.CONFIG['ADX_THRESHOLD']}**")

        # CSV 경로(csv_path)를 반환받음                       
        summary, _, csv_path = await bot.loop.run_in_executor(
            None, lambda: backtest_runner.run_walk_forward(symbol, test_days, train_days, step_days)
        )

        if summary:
            await ctx.send(summary)

        # [NEW] CSV 파일 전송 및 삭제                         
        if csv_path and os.path.exists(csv_path):
            try:
                file = discord.File(csv_path, filename=f"TradeLog_{symbol.replace('/','')}.csv")
                await ctx.send(file=file)
            except Exception as e:
                await ctx.send(f"❌ 로그 파일 전송 실패: {e}")
            finally:
                # 전송 후 파일 삭제 (서버 용량 관리)          
                os.remove(csv_path)
                print(f"🗑️ 임시 파일 삭제 완료: {csv_path}")

    finally:
        is_backtesting = False

@bot.command(name="전체백테", aliases=["abt"])
async def cmd_batch_backtest(ctx, test_days: int = 30, train_days: int = 365):
    global is_backtesting
    if not is_admin(ctx): return

    if is_backtesting:
        await ctx.send("⏳ **이미 다른 백테스팅이 실행 중입니다.**")
        return

    is_backtesting = True
    try:
        await ctx.send(f"⏳ **전체 포트폴리오** WFA 시뮬레이션\n" 
                       f"🎯 검증 기간: 최근 **{test_days}일**\n" 
                       f"📚 학습 기간: **{train_days}일**")

        report = await bot.loop.run_in_executor(None, lambda: backtest_runner.run_batch_backtest(train_days, test_days))

        if len(report) > 1900:
            for i in range(0, len(report), 1900):
                await ctx.send(report[i:i+1900])
        else:
            await ctx.send(report)
    finally:
        is_backtesting = False

# 나머지 명령어 (!청산, !상태 등) 및 bot.run은 기존과 동일하게 유지
@bot.command(name="청산", aliases=["close"])
async def cmd_close(ctx, symbol: str):
    if not is_admin(ctx): return
    await ctx.send(f"⚠️ **{symbol}** 안전한 강제 청산 시도 중...")
    result = await trader.safe_force_close(symbol)
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

@bot.command(name="실매매시작")
async def start_real(ctx):
    if not is_admin(ctx): return
    msg = trader.set_mode('REAL')
    await ctx.send(f"🚨 **주의!** 실제 자산이 투입됩니다.\n{msg}")

@bot.command(name="모의매매시작")
async def start_paper(ctx):
    if not is_admin(ctx): return
    msg = trader.set_mode('PAPER')
    await ctx.send(f"🧪 가상 트레이딩 모드입니다.\n{msg}")

@bot.command(name="정지", aliases=["stop"])
async def stop_bot(ctx):
    if not is_admin(ctx): return
    msg = trader.set_mode('OFF')
    await ctx.send(f"🛑 모든 매매 활동을 중단합니다.\n{msg}")

bot.run(TOKEN)