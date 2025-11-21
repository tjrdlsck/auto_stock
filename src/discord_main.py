import matplotlib
matplotlib.use('Agg') # 서버 환경에서 GUI 창 안 뜨게 설정

import os
import discord
import asyncio
import requests  # [NEW] Heartbeat용
from discord.ext import commands, tasks
from dotenv import load_dotenv

# ---------------------------------------------------------
# [1] 사용자 모듈 임포트
# ---------------------------------------------------------
from config import CONFIG, DATA_DIR
from trader import BinanceTrader
import backtest_runner
from data_layer.data_handler import MultiSymbolLoader  # 데이터 수집기
from alpha_layer.brain import Brain                     # AI 학습기

# 환경변수 로드
load_dotenv()
TOKEN = os.getenv("DISCORD_BOT_TOKEN")
ADMIN_ID = int(os.getenv("DISCORD_ADMIN_ID"))

# [설정] Healthchecks.io에서 발급받은 Ping URL
HEARTBEAT_URL = "https://hc-ping.com/800eca18-a428-4a27-87ee-071a92eb0298"

# 봇 설정
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

# 트레이더 인스턴스 생성
trader = BinanceTrader()

# ---------------------------------------------------------
# [2] 메신저 브릿지 (Trader -> Discord 알림)
# ---------------------------------------------------------
def get_notification_channel():
    """알림을 보낼 첫 번째 텍스트 채널 찾기"""
    for guild in bot.guilds:
        if guild.text_channels:
            return guild.text_channels[0]
    return None

async def send_discord_alert(symbol, side, price, qty, msg_type, image_path=None, mention_everyone=False):
    channel = get_notification_channel()
    if channel:
        color = 0x00ff00 if side.lower() == 'buy' else 0xff0000
        emoji = "🚀" if side.lower() == 'buy' else "📉"
        
        embed = discord.Embed(title=f"{emoji} BINANCE {side.upper()} {msg_type}", color=color)
        embed.add_field(name="코인", value=f"`{symbol}`", inline=True)
        embed.add_field(name="가격", value=f"`${price:,.4f}`", inline=True)
        embed.add_field(name="수량", value=f"`{qty}`", inline=True)
        embed.set_footer(text="QuantBot V13 (Portfolio & Auto-Learn)")

        content = "@everyone" if mention_everyone else None

        if image_path and os.path.exists(image_path):
            file = discord.File(image_path, filename="backtest_result.png")
            embed.set_image(url="attachment://backtest_result.png")
            await channel.send(embed=embed, file=file, content=content)
        else:
            await channel.send(embed=embed, content=content)

def messenger_bridge(symbol, side, price, qty, msg_type, image_path=None, mention_everyone=False):
    """Trader에서 호출하는 동기 함수 -> 비동기 변환"""
    if bot.loop.is_running():
        bot.loop.create_task(send_discord_alert(symbol, side, price, qty, msg_type, image_path, mention_everyone))

# 트레이더에 메신저 연결
trader.messenger = messenger_bridge

def is_admin(ctx):
    return ctx.author.id == ADMIN_ID

# ---------------------------------------------------------
# [3] 자동화 루프 (Scheduler)
# ---------------------------------------------------------

@tasks.loop(minutes=1)
async def heartbeat_loop():
    """매분 외부 감시 서버(Healthchecks.io)에 생존 신호 전송"""
    try:
        # 비동기 환경에서 requests는 차단을 유발할 수 있으므로 executor 사용
        await bot.loop.run_in_executor(None, requests.get, HEARTBEAT_URL)
        # 로그가 너무 많으면 아래 print는 주석 처리해도 됨
        # print("💓 [Heartbeat] 생존 신호 전송 완료")
    except Exception as e:
        print(f"⚠️ Heartbeat 전송 실패: {e}")

@tasks.loop(hours=1)
async def hourly_trade_loop():
    """매 시간 정각마다 매매 로직 수행"""
    print(f"\n⏰ [Hourly Loop] {backtest_runner.datetime.now()} - 매매 로직 시작")
    try:
        result = await trader.run_logic()
        print(f"✅ {result}")
    except Exception as e:
        print(f"❌ 매매 루프 에러: {e}")

@tasks.loop(hours=CONFIG['RETRAIN_INTERVAL_HOURS'])
async def auto_retraining_loop():
    """정기 데이터 수집 및 모델 재학습 (Blocking 방지 적용)"""
    print(f"\n🎓 [Auto-Retrain] 시스템 재학습 프로세스 시작")
    
    channel = get_notification_channel()
    if channel:
        await channel.send("🔄 **[시스템]** 일일 데이터 업데이트 및 AI 모델 재학습을 시작합니다...")

    try:
        # 1. 데이터 수집 (오래 걸리므로 Executor에서 실행)
        loader = MultiSymbolLoader()
        # run_pipeline은 동기 함수이므로 run_in_executor로 래핑
        await bot.loop.run_in_executor(None, loader.run_pipeline)
        
        # 2. 모델 학습
        brain = Brain()
        await bot.loop.run_in_executor(None, brain.run_training)
        
        print("✅ 정기 재학습 완료")
        if channel:
            await channel.send("✅ **[시스템]** AI 모델 재학습 완료! 최신 시장 트렌드가 반영되었습니다.")
            
    except Exception as e:
        print(f"❌ 재학습 중 치명적 오류: {e}")
        if channel:
            await channel.send(f"⚠️ **[오류]** 재학습 실패: {e}")

@auto_retraining_loop.before_loop
async def before_retraining():
    await bot.wait_until_ready()

# ---------------------------------------------------------
# [4] 봇 이벤트 및 명령어
# ---------------------------------------------------------

@bot.event
async def on_ready():
    # [수정됨] 중복된 on_ready를 하나로 합침
    print(f'🤖 디스코드 봇 로그인 성공: {bot.user}')
    print(f'⏳ 스케줄러 가동 중... (매매: 1시간, 재학습: {CONFIG["RETRAIN_INTERVAL_HOURS"]}시간, 하트비트: 1분)')
    
    if not hourly_trade_loop.is_running():
        hourly_trade_loop.start()
    
    if not auto_retraining_loop.is_running():
        auto_retraining_loop.start()
        
    if not heartbeat_loop.is_running():
        heartbeat_loop.start()

@bot.event
async def on_message(message):
    if message.author.bot: return

    if message.content == "!":
        embed = discord.Embed(title="🤖 퀀트 봇 V13 명령어", description="Portfolio & Data Isolation Applied", color=0x3498db)
        embed.add_field(name="🕹️ 제어", value="`!모의매매시작`, `!실매매시작`, `!정지`, `!상태`", inline=False)
        embed.add_field(name="🚨 긴급", value="`!청산 [코인]` (안전 강제 청산)", inline=False)
        embed.add_field(name="📊 백테스트", value="`!백테 [코인]`, `!전체백테`", inline=False)
        embed.set_footer(text="주의: 실매매 전환 시 DB가 격리되어 포지션 목록이 초기화됩니다.")
        await message.channel.send(embed=embed)
        return

    await bot.process_commands(message)

# --- 매매 제어 명령어 ---

@bot.command(name="상태", aliases=["status"])
async def status(ctx):
    if not is_admin(ctx): return
    
    free_usdt, total_usdt = await trader.get_balance()
    positions = await trader.get_positions()
    mode = trader.mode
    
    # 모드에 따른 아이콘
    mode_icon = "🟢 REAL" if mode == 'REAL' else ("🧪 PAPER" if mode == 'PAPER' else "🔴 OFF")
    
    embed = discord.Embed(title=f"📊 퀀트 봇 상태 ({mode_icon})", color=0x3498db)
    embed.add_field(name="💰 자산 현황", value=f"총액: **${total_usdt:,.2f}**\n가용: ${free_usdt:,.2f}", inline=False)
    
    if positions:
        pos_str = ""
        for p in positions:
            pnl = p.get('unrealizedPnl', 0.0)
            icon = "🟢" if pnl >= 0 else "🔴"
            pos_str += f"**{p['symbol']}** ({p['side']} x{p['leverage']})\n{icon} ${pnl:.2f} (Entry: ${p['entryPrice']:.4f})\n\n"
        embed.add_field(name=f"📈 보유 포지션 ({len(positions)}/{CONFIG['MAX_OPEN_POSITIONS']})", value=pos_str, inline=False)
    else:
        embed.add_field(name="📈 보유 포지션", value="보유 중인 포지션 없음", inline=False)
        
    await ctx.send(embed=embed)

@bot.command(name="실매매시작")
async def start_real(ctx):
    if not is_admin(ctx): return
    msg = trader.set_mode('REAL')
    await ctx.send(f"🚨 **[경고] 실전 매매 모드 전환**\n실제 자산이 투입됩니다. (DB 격리: PAPER 데이터 숨김)\n{msg}")

@bot.command(name="모의매매시작")
async def start_paper(ctx):
    if not is_admin(ctx): return
    msg = trader.set_mode('PAPER')
    await ctx.send(f"🧪 **모의 매매 모드 전환**\n가상 자금으로 테스트합니다. (DB 격리: REAL 데이터 숨김)\n{msg}")

@bot.command(name="정지", aliases=["stop"])
async def stop_bot(ctx):
    if not is_admin(ctx): return
    msg = trader.set_mode('OFF')
    await ctx.send(f"🛑 **시스템 정지**\n모든 신규 진입 및 청산 로직이 중단됩니다.\n{msg}")

@bot.command(name="청산", aliases=["close"])
async def cmd_close(ctx, symbol: str):
    if not is_admin(ctx): return
    await ctx.send(f"⚠️ **{symbol}** 강제 청산 명령 수신. 처리 중...")
    result = await trader.safe_force_close(symbol)
    await ctx.send(result)

# --- 백테스팅 명령어 ---

is_backtesting = False

@bot.command(name="백테", aliases=["bt"])
async def cmd_backtest(ctx, symbol: str = "BTC/USDT", test_days: int = 30, train_days: int = 365):
    global is_backtesting
    if not is_admin(ctx): return

    if is_backtesting:
        await ctx.send("⏳ **다른 백테스팅이 진행 중입니다.**")
        return

    is_backtesting = True
    try:
        step_days = 30 if test_days >= 30 else test_days
        
        # [NEW] 1. 단일 심볼 데이터 최신화
        await ctx.send(f"📥 **{symbol}** 최신 데이터 업데이트 중...")
        
        def update_single_symbol():
            loader = MultiSymbolLoader()
            # fetch_ohlcv는 데이터만 가져오므로, 피처 엔지니어링 후 저장이 필요함
            df, file_path = loader.fetch_ohlcv(symbol, days=CONFIG['FETCH_DAYS'])
            if not df.empty:
                df = loader.add_features(df)
                df.to_csv(file_path)
                return True
            return False

        success = await bot.loop.run_in_executor(None, update_single_symbol)
        
        if not success:
            await ctx.send(f"❌ **{symbol}** 데이터 업데이트 실패. 백테스팅을 중단합니다.")
            return

        # [NEW] 2. 백테스팅 수행
        await ctx.send(f"⏳ **{symbol}** WFA 백테스트 시작 (검증: {test_days}일, 학습: {train_days}일)")

        summary, _, csv_path = await bot.loop.run_in_executor(
            None, lambda: backtest_runner.run_walk_forward(symbol, test_days, train_days, step_days)
        )

        if summary: await ctx.send(summary)
        
        # 로그 파일 전송 및 삭제
        if csv_path and os.path.exists(csv_path):
            try:
                file = discord.File(csv_path, filename=f"Log_{symbol.replace('/','')}.csv")
                await ctx.send(file=file)
                os.remove(csv_path)
            except: pass
            
    except Exception as e:
        await ctx.send(f"❌ 에러 발생: {e}")
    finally:
        is_backtesting = False

@bot.command(name="전체백테", aliases=["abt"])
async def cmd_batch_backtest(ctx, test_days: int = 30, train_days: int = 365):
    global is_backtesting
    if not is_admin(ctx): return
    if is_backtesting: return await ctx.send("⏳ **진행 중입니다.**")

    is_backtesting = True
    try:
        # [NEW] 1. 전체 데이터 최신화
        await ctx.send(f"📥 **전체 포트폴리오 ({len(CONFIG['SYMBOLS'])}개)** 데이터 최신화 중...")
        
        loader = MultiSymbolLoader()
        # run_pipeline은 모든 심볼을 수집하고 저장까지 자동으로 수행함
        await bot.loop.run_in_executor(None, loader.run_pipeline)
        
        # [NEW] 2. 일괄 시뮬레이션 수행
        await ctx.send(f"⏳ 시뮬레이션 시작... (시간이 다소 소요됩니다)")
        report = await bot.loop.run_in_executor(None, lambda: backtest_runner.run_batch_backtest(train_days, test_days))
        
        if len(report) > 1900:
            for i in range(0, len(report), 1900):
                await ctx.send(report[i:i+1900])
        else:
            await ctx.send(report)
            
    except Exception as e:
        await ctx.send(f"❌ 에러 발생: {e}")
    finally:
        is_backtesting = False

bot.run(TOKEN)