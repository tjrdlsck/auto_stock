from dataclasses import dataclass, field
from datetime import datetime, timedelta  # <--- [수정됨] timedelta 추가
from enum import Enum
from typing import Optional

# =========================================================
# 1. Enums (상수 정의)
# =========================================================

class OrderType(Enum):
    MARKET = "MARKET"       # 시장가
    LIMIT = "LIMIT"         # 지정가
    STOP_MARKET = "STOP"    # 스탑(시장가) - 손절/돌파용

class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"

class OrderStatus(Enum):
    CREATED = "CREATED"     # 생성됨 (아직 브로커 제출 전)
    SUBMITTED = "SUBMITTED" # 브로커에 제출됨 (대기 중)
    COMPLETED = "COMPLETED" # 체결 완료
    CANCELED = "CANCELED"   # 취소됨
    REJECTED = "REJECTED"   # 거부됨 (증거금 부족 등)

# =========================================================
# 2. Data Classes (데이터 객체)
# =========================================================

@dataclass
class Order:
    """
    개별 주문 정보를 담는 객체
    """
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None       # 시장가 주문시 None 가능
    stop_price: Optional[float] = None  # Stop 주문시 트리거 가격
    
    # 상태 관리
    status: OrderStatus = OrderStatus.CREATED
    created_at: datetime = field(default_factory=datetime.now)
    id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d%H%M%S%f"))

    def __repr__(self):
        p = f"@{self.price}" if self.price else "MKT"
        return f"<Order {self.side.value} {self.symbol} {self.quantity} {p} ({self.status.value})>"


@dataclass
class Trade:
    """
    완료된 거래(진입~청산) 내역을 담는 객체
    기존 trade_log 딕셔너리를 대체합니다.
    """
    symbol: str
    entry_time: datetime
    exit_time: datetime
    side: OrderSide      # 진입 방향 (BUY=Long, SELL=Short)
    quantity: float
    entry_price: float
    exit_price: float
    
    gross_pnl: float = 0.0 # 수수료 제외 전 손익
    net_pnl: float = 0.0   # 수수료 포함 최종 손익
    commission: float = 0.0
    
    exit_reason: str = "NORMAL" # STOP_LOSS, TAKE_PROFIT, LIQUIDATION, SIGNAL ...
    
    # 성과 분석용 추가 필드
    pnl_pct: float = 0.0        # 수익률 (%)
    holding_period: timedelta = field(default_factory=timedelta) # <--- 이제 에러가 나지 않습니다

@dataclass
class Position:
    """
    현재 보유 중인 포지션 정보를 담는 객체
    격리 마진(Isolated Margin) 및 청산가(Liq Price) 로직을 포함합니다.
    """
    symbol: str
    side: OrderSide
    quantity: float
    entry_price: float
    entry_time: datetime
    
    # 리스크 관리 필드 (기존 RiskManager 로직 대응)
    leverage: float = 1.0
    initial_margin: float = 0.0  # 격리 증거금 (진입 시 할당된 금액)
    liq_price: float = 0.0       # 청산가
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    
    # 상태 추적
    current_price: float = 0.0 # 현재가 (평가용)
    
    def update_price(self, price: float):
        self.current_price = price

    @property
    def unrealized_pnl(self) -> float:
        """미실현 손익 계산 (단순 차익)"""
        if self.side == OrderSide.BUY:
            return (self.current_price - self.entry_price) * self.quantity
        else:
            return (self.entry_price - self.current_price) * self.quantity

    @property
    def unrealized_pnl_pct(self) -> float:
        """ROE (Return on Equity) 계산"""
        if self.initial_margin == 0: return 0.0
        return (self.unrealized_pnl / self.initial_margin) * 100

    def __repr__(self):
        return (f"<Position {self.symbol} {self.side.value} x{self.quantity} "
                f"Entry:{self.entry_price:.2f} Now:{self.current_price:.2f} "
                f"PnL:{self.unrealized_pnl:.2f}>")