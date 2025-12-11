from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any
import pandas as pd

@dataclass
class TradeSignal:
    """
    전략이 백테스터(또는 봇)에게 전달하는 매매 신호 표준 포맷
    """
    timestamp: Any          # 신호 발생 시간
    action: str             # "BUY", "SELL", "HOLD"
    entry_price: float      # 진입 목표 가격 (또는 현재가)
    stop_loss: Optional[float] = None   # 손절 가격
    take_profit: Optional[float] = None # 익절 가격
    quantity: Optional[float] = None    # (선택) 전략이 직접 수량을 제안할 경우
    reason: str = ""        # 로그용 사유 (디버깅)
    trailing_stop_multiplier: Optional[float] = None  # [추가됨] 트레일링 스탑 배수 (None이면 사용 안 함)
    def __str__(self):
        return f"[{self.action}] @ {self.entry_price} (SL: {self.stop_loss}, TP: {self.take_profit}) | {self.reason}"

class BaseStrategy(ABC):
    """
    모든 트레이딩 전략의 부모 클래스 (Interface)
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}

    @abstractmethod
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        전체 데이터셋에 대해 필요한 보조지표를 미리 계산합니다.
        :param df: OHLCV 데이터프레임
        :return: 지표가 추가된 데이터프레임
        """
        pass

    @abstractmethod
    def generate_signal(self, data: Any) -> TradeSignal:
        """
        현재 시점의 데이터를 기준으로 매매 판단을 내립니다.
        :param data: 현재 시점의 데이터 (DataFrame Row, NamedTuple 등)
        :return: TradeSignal 객체
        """
        pass
