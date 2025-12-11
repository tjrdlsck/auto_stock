from abc import ABC, abstractmethod
from typing import Optional
from .objects import Order, OrderType, OrderSide, Position
from .broker import Broker
from .datafeed import DataFeed

class BaseStrategy(ABC):
    """
    모든 매매 전략의 부모 클래스.
    사용자는 이 클래스를 상속받아 initialize()와 next()를 구현해야 합니다.
    """
    def __init__(self, broker: Broker, feed: DataFeed, params: dict = None):
        self.broker = broker
        self.feed = feed
        self.params = params if params else {}
        
        # 편의를 위한 데이터 단축 속성
        self.data = self.feed
        
        # 사용자 초기화 함수 호출
        self.initialize()

    def initialize(self):
        """
        지표(Indicators) 설정 등 초기화 로직을 오버라이딩하여 구현합니다.
        """
        pass

    @abstractmethod
    def next(self):
        """
        매 틱(Tick)마다 호출되는 메인 로직.
        self.data.close 등으로 현재 데이터에 접근하고,
        self.buy(), self.sell() 등으로 주문을 냅니다.
        """
        pass

    # =========================================================
    # 주문 편의 메서드 (Wrapper)
    # =========================================================

    # [수정] immediate 파라미터 추가
    def buy(self, 
            quantity: float, 
            price: Optional[float] = None, 
            stop_price: Optional[float] = None,
            symbol: Optional[str] = None,
            immediate: bool = False) -> Order:
        """매수 주문 전송 (immediate=True시 즉시 체결)"""
        return self._submit(OrderSide.BUY, quantity, price, stop_price, symbol, immediate)

    # [수정] immediate 파라미터 추가
    def sell(self, 
             quantity: float, 
             price: Optional[float] = None, 
             stop_price: Optional[float] = None,
             symbol: Optional[str] = None,
             immediate: bool = False) -> Order:
        """매도 주문 전송 (immediate=True시 즉시 체결)"""
        return self._submit(OrderSide.SELL, quantity, price, stop_price, symbol, immediate)

    # [수정] price, immediate 파라미터 추가
    def close(self, symbol: str = None, price: float = None, immediate: bool = False):
        """현재 포지션 청산"""
        target_symbol = symbol if symbol else self.params.get('symbol')
        if target_symbol:
            pos = self.broker.positions.get(target_symbol)
            if pos:
                side = OrderSide.SELL if pos.side == OrderSide.BUY else OrderSide.BUY
                p = price if price else self.feed.close
                # 청산은 수량 전체
                self._submit(side, pos.quantity, p, None, target_symbol, immediate)

    # [수정] immediate 인자 처리 및 submit_order 호출
    def _submit(self, side, quantity, price, stop_price, symbol, immediate):
        target_symbol = symbol if symbol else self.params.get('symbol', 'UNKNOWN')
        
        # 주문 타입 결정
        if stop_price:
            o_type = OrderType.STOP_MARKET
        elif price and not immediate: # 지정가 주문
            o_type = OrderType.LIMIT
        else: # 시장가 주문 (또는 즉시 체결)
            o_type = OrderType.MARKET
            
        order = Order(
            symbol=target_symbol,
            side=side,
            order_type=o_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price
        )
        
        # 브로커에게 플래그 전달
        self.broker.submit_order(order, immediate=immediate)
        return order

    # =========================================================
    # 정보 조회 메서드
    # =========================================================
    
    @property
    def position(self) -> Optional[Position]:
        """현재 기본 심볼의 포지션 객체 반환 (없으면 None)"""
        sym = self.params.get('symbol')
        return self.broker.positions.get(sym)
    
    @property
    def equity(self) -> float:
        """현재 총 자산 평가액"""
        return self.broker.get_total_equity()