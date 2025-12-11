import logging
import pandas as pd
from typing import Type, Dict
from .broker import Broker
from .datafeed import DataFeed
from .strategy import BaseStrategy

class Engine:
    """
    백테스팅 시뮬레이션을 구동하는 메인 컨트롤러.
    데이터 주입 -> 전략 설정 -> 실행(Run) 순서로 사용합니다.
    """
    def __init__(self, initial_cash: float = 10000.0):
        self.initial_cash = initial_cash
        self.broker = Broker(initial_cash=initial_cash)
        self.feed = None
        self.strategy_cls = None
        self.strategy_params = {}
        
        # 로깅 설정 (기본은 INFO)
        logging.basicConfig(level=logging.INFO, format='%(message)s')
        self.logger = logging.getLogger("Engine")

    def add_data(self, df: pd.DataFrame):
        """데이터프레임을 DataFeed로 변환하여 등록"""
        self.feed = DataFeed(df)

    def add_strategy(self, strategy_cls: Type[BaseStrategy], **kwargs):
        """실행할 전략 클래스와 파라미터 등록"""
        self.strategy_cls = strategy_cls
        self.strategy_params = kwargs

    def run(self) -> Dict:
        """
        시뮬레이션 메인 루프 실행
        """
        if not self.feed or not self.strategy_cls:
            raise ValueError("Data and Strategy must be added before running.")

        # 1. 초기화
        self.feed.reset()
        self.broker = Broker(initial_cash=self.initial_cash) # 브로커 리셋
        
        # 전략 인스턴스 생성 (브로커와 피드 주입)
        strategy = self.strategy_cls(
            broker=self.broker, 
            feed=self.feed, 
            params=self.strategy_params
        )
        
        self.logger.info(f"🚀 Simulation Started: {self.strategy_cls.__name__}")
        self.logger.info(f"   Initial Balance: ${self.initial_cash:,.2f}")

        # 2. 시간 루프 (Time Loop)
        # feed.next()가 True인 동안 계속 진행 (데이터 끝날 때까지)
        while self.feed.next():
            # A. 브로커 업데이트 (주문 체결 확인, 청산 감시)
            self.broker.match(self.feed)
            
            # B. 전략 실행 (매수/매도 판단)
            strategy.next()

        # 3. 종료 및 결과 집계
        final_equity = self.broker.get_total_equity()
        roi = ((final_equity - self.initial_cash) / self.initial_cash) * 100
        
        self.logger.info(f"🏁 Simulation Finished.")
        self.logger.info(f"   Final Balance: ${final_equity:,.2f} (ROI: {roi:.2f}%)")
        
        return self._generate_report(final_equity, roi)

    def _generate_report(self, final_equity, roi):
        """결과 딕셔너리 생성"""
        trades = self.broker.trades
        
        # 승률 계산
        wins = [t for t in trades if t.net_pnl > 0]
        win_rate = (len(wins) / len(trades)) * 100 if trades else 0.0
        
        # MDD 계산 (Equity Curve가 필요하다면 Broker에서 기록해야 함. 여기선 약식)
        # Broker에 equity history 기능이 추가되면 더 정밀해질 수 있음.
        
        return {
            "initial_balance": self.initial_cash,
            "final_balance": final_equity,
            "roi_pct": roi,
            "total_trades": len(trades),
            "win_rate_pct": win_rate,
            "trades": trades # 거래 내역 객체 리스트
        }