import logging
import pandas as pd
from typing import Type, Dict
from .broker import Broker
from .datafeed import DataFeed
from .strategy import BaseStrategy

class Engine:
    """
    백테스팅 시뮬레이션을 구동하는 메인 컨트롤러.
    """
    # [수정] verbose 파라미터 추가
    def __init__(self, initial_cash: float = 10000.0, verbose: bool = True):
        self.initial_cash = initial_cash
        self.verbose = verbose  # 로그 출력 여부 제어
        self.broker = Broker(initial_cash=initial_cash)
        self.feed = None
        self.strategy_cls = None
        self.strategy_params = {}
        
        # 로깅 설정
        self.logger = logging.getLogger("Engine")
        # verbose가 False면 로그 레벨을 높여서 INFO 출력을 막음
        if not self.verbose:
            self.logger.setLevel(logging.WARNING)
        else:
            self.logger.setLevel(logging.INFO)

    def add_data(self, df: pd.DataFrame):
        self.feed = DataFeed(df)

    def add_strategy(self, strategy_cls: Type[BaseStrategy], **kwargs):
        self.strategy_cls = strategy_cls
        self.strategy_params = kwargs

    def run(self) -> Dict:
        if not self.feed or not self.strategy_cls:
            raise ValueError("Data and Strategy must be added before running.")

        # 1. 초기화
        self.feed.reset()
        self.broker = Broker(initial_cash=self.initial_cash) 
        
        strategy = self.strategy_cls(
            broker=self.broker, 
            feed=self.feed, 
            params=self.strategy_params
        )
        
        if self.verbose:
            self.logger.info(f"🚀 Simulation Started: {self.strategy_cls.__name__}")
            self.logger.info(f"   Initial Balance: ${self.initial_cash:,.2f}")

        # 2. 시간 루프
        while self.feed.next():
            self.broker.match(self.feed)
            strategy.next()

        # 3. 종료 및 결과 집계
        final_equity = self.broker.get_total_equity()
        roi = ((final_equity - self.initial_cash) / self.initial_cash) * 100
        
        if self.verbose:
            self.logger.info(f"🏁 Simulation Finished.")
            self.logger.info(f"   Final Balance: ${final_equity:,.2f} (ROI: {roi:.2f}%)")
        
        return self._generate_report(final_equity, roi)

    def _generate_report(self, final_equity, roi):
        trades = self.broker.trades
        wins = [t for t in trades if t.net_pnl > 0]
        win_rate = (len(wins) / len(trades)) * 100 if trades else 0.0
        
        return {
            "initial_balance": self.initial_cash,
            "final_balance": final_equity,
            "roi_pct": roi,
            "total_trades": len(trades),
            "win_rate_pct": win_rate,
            "trades": trades 
        }