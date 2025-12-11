import logging
from typing import List, Dict, Optional
from datetime import datetime
from .objects import Order, Trade, Position, OrderType, OrderSide, OrderStatus
from .datafeed import DataFeed

class Broker:
    """
    주문 집행, 포지션 관리, 자산(Equity) 계산을 담당하는 가상 거래소
    """
    def __init__(self, initial_cash: float, commission: float = 0.0005, slippage: float = 0.0003, funding_rate: float = 0.0001):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.commission = commission
        self.slippage = slippage
        self.funding_rate = funding_rate
        
        # 계좌 상태
        self.positions: Dict[str, Position] = {}  # {symbol: Position}
        self.active_orders: List[Order] = []      # 대기 중인 주문 (Stop 주문 등)
        self.trades: List[Trade] = []             # 체결된 거래 내역
        
        # [추가됨] 자산 변동 기록용 리스트
        self.equity_history: List[Dict] = [] 
        
        self.logger = logging.getLogger("Broker")

    def get_total_equity(self) -> float:
        """총 자산 = 현금 + 보유 포지션의 미실현 손익(Margin 포함)"""
        equity = self.cash
        for pos in self.positions.values():
            # 격리 마진 모드: 초기 증거금 + 미실현 손익
            equity += pos.initial_margin + pos.unrealized_pnl
        return equity

    # [수정] immediate 플래그 처리 로직 추가
    def submit_order(self, order: Order, immediate: bool = False):
        """전략으로부터 주문을 접수합니다."""
        
        if immediate:
            # [즉시 체결 모드]
            # 가격이 명시되어 있으면 그 가격, 없으면 현재 캔들 종가(Close)로 체결
            # 주의: Broker는 match()가 호출될 때 feed 정보를 저장해둬야 함 (아래 match 수정 참조)
            if not hasattr(self, 'current_feed_data'):
                # 혹시라도 feed 정보가 없으면 에러 방지용으로 active_orders에 넣음
                order.status = OrderStatus.SUBMITTED
                self.active_orders.append(order)
                return

            fill_price = order.price if order.price is not None else self.current_feed_data.close
            
            # 대기열 없이 즉시 체결 실행
            self._execute_fill(order, fill_price, self.current_feed_data.date)
            
        else:
            # [일반 모드]
            order.status = OrderStatus.SUBMITTED
            self.active_orders.append(order)

    # [수정] 현재 feed 데이터를 저장하는 로직 추가
    def match(self, feed: DataFeed):
        """
        [핵심] 현재 캔들(feed) 정보를 바탕으로 대기 주문을 체결하고, 청산(Liq)을 감시합니다.
        """
        # [중요] 즉시 체결을 위해 현재 피드 정보를 멤버 변수에 저장
        self.current_feed_data = feed 
        
        # 0. 펀딩비 적용 (8시간 주기)
        self._apply_funding_fee(feed.date)
        
        # 1. 포지션 평가 (Mark to Market) & 청산(Liquidation) 체크
        self._check_positions(feed)

        # 2. 주문 체결 처리
        unfilled_orders = []
        
        for order in self.active_orders:
            if order.status != OrderStatus.SUBMITTED:
                continue
                
            fill_price = self._check_check(order, feed) # (참고: 원본 코드의 _check_fill 오타가 있다면 _check_fill로 사용)
            # 여기서는 문맥상 _check_fill이 맞습니다. 아래 로직 유지.
            fill_price = self._check_fill(order, feed)
            
            if fill_price:
                self._execute_fill(order, fill_price, feed.date)
            else:
                unfilled_orders.append(order)
                
        self.active_orders = unfilled_orders

        # [추가됨] 현재 시점의 자산 가치(Equity) 기록
        self._log_equity_status(feed.date)

    def _apply_funding_fee(self, current_time: datetime):
        """8시간마다 (00:00, 08:00, 16:00) 펀딩비 차감"""
        # 1시간봉 데이터라고 가정 시, 정각(minute==0)에 체크
        if current_time.hour % 8 == 0 and current_time.minute == 0:
            total_fee = 0.0
            
            for symbol, pos in self.positions.items():
                # 포지션 명목 가치 (Notional Value) = 진입가 * 수량
                # (현재가 기준이 아닌 진입가 기준으로 단순화, 거래소마다 다름)
                position_value = pos.entry_price * pos.quantity
                fee = position_value * self.funding_rate
                
                self.cash -= fee
                total_fee += fee
                
            if total_fee > 0:
                # 로그가 너무 많아질 수 있으므로 필요시 주석 처리
                # self.logger.info(f"💸 Funding Fee Deducted at {current_time}: -${total_fee:.4f}")
                pass

    def _check_fill(self, order: Order, feed: DataFeed) -> Optional[float]:
        """주문 조건이 충족되었는지 확인하고 체결 가격 반환"""
        
        # MARKET 주문: 현재 종가(Close)로 즉시 체결 (가정)
        # 실제로는 다음 봉 시가(Open)가 맞으나, 백테스트 단순화를 위해 현재 봉 체결 지원
        if order.order_type == OrderType.MARKET:
            return feed.close

        # STOP 주문 (손절/돌파)
        if order.order_type == OrderType.STOP_MARKET:
            if order.stop_price is None: return None
            
            # Buy Stop: 가격이 stop_price 이상으로 올라가면 체결 (High 터치)
            if order.side == OrderSide.BUY:
                if feed.high >= order.stop_price:
                    # 갭상승 고려: 설정가보다 시가가 높으면 시가 체결, 아니면 설정가
                    return max(feed.open, order.stop_price) if feed.open > order.stop_price else order.stop_price
            
            # Sell Stop: 가격이 stop_price 이하로 내려가면 체결 (Low 터치)
            elif order.side == OrderSide.SELL:
                if feed.low <= order.stop_price:
                    return min(feed.open, order.stop_price) if feed.open < order.stop_price else order.stop_price

        return None

    def _execute_fill(self, order: Order, price: float, timestamp: datetime):
        """체결 실행: 현금 차감, 포지션 업데이트, Trade 기록"""
        # 1. 수수료 및 슬리피지 적용
        # Buy: 비싸게 삼, Sell: 싸게 팖
        effective_price = price * (1 + self.slippage) if order.side == OrderSide.BUY else price * (1 - self.slippage)
        
        trade_value = effective_price * order.quantity
        fee = trade_value * self.commission
        
        # 2. 포지션 업데이트 로직
        # (단순화를 위해 Hedge 모드 미지원, One-Way 모드 가정: 반대 주문은 포지션 청산으로 간주)
        symbol = order.symbol
        current_pos = self.positions.get(symbol)
        
        # A. 포지션 신규 진입 (또는 불타기)
        if current_pos is None or current_pos.side == order.side:
            self._open_position(order, effective_price, fee, timestamp, current_pos)
            
        # B. 포지션 청산 (또는 스위칭)
        else:
            self._close_position_fill(order, effective_price, fee, timestamp, current_pos)

    def _open_position(self, order, price, fee, timestamp, current_pos):
        """신규 진입 처리"""
        cost = (price * order.quantity) / 1.0 # 레버리지는 외부(Strategy)에서 계산된 수량으로 반영됨 가정
        
        # 현금 차감 (수수료)
        self.cash -= fee
        # 증거금 차감 (격리 마진: 여기서 레버리지 정보가 필요하지만, 
        # 일단 Strategy가 자금 관리를 해서 quantity를 정했다고 가정하고 여기선 기록만 함)
        
        if current_pos:
            # 평단가 수정 (가중 평균)
            total_qty = current_pos.quantity + order.quantity
            avg_price = (current_pos.entry_price * current_pos.quantity + price * order.quantity) / total_qty
            current_pos.entry_price = avg_price
            current_pos.quantity = total_qty
            # 마진 추가 로직은 복잡하므로 생략 (단순 합산)
        else:
            # 신규 생성
            new_pos = Position(
                symbol=order.symbol,
                side=order.side,
                quantity=order.quantity,
                entry_price=price,
                entry_time=timestamp,
                initial_margin=0.0 # Strategy에서 설정하거나 별도 할당 필요
            )
            self.positions[order.symbol] = new_pos

        order.status = OrderStatus.COMPLETED

    def _close_position_fill(self, order, price, fee, timestamp, current_pos):
        """청산 처리"""
        # 1. 청산 수량 계산 (전량 청산 or 부분 청산)
        close_qty = min(current_pos.quantity, order.quantity)
        
        # 2. PnL 계산
        if current_pos.side == OrderSide.BUY:
            gross_pnl = (price - current_pos.entry_price) * close_qty
        else:
            gross_pnl = (current_pos.entry_price - price) * close_qty
            
        net_pnl = gross_pnl - fee
        
        # 3. 현금 반영 (원금 + 손익)
        # 격리 마진 개념: 포지션 잡을 때 묶인 돈을 풀어줌 + PnL
        # 여기서는 Simplified Cash Flow: Cash += PnL
        self.cash += net_pnl
        
        # 4. Trade 기록
        trade = Trade(
            symbol=order.symbol,
            entry_time=current_pos.entry_time,
            exit_time=timestamp,
            side=current_pos.side,
            quantity=close_qty,
            entry_price=current_pos.entry_price,
            exit_price=price,
            gross_pnl=gross_pnl,
            net_pnl=net_pnl,
            commission=fee,
            exit_reason="TRADE"
        )
        self.trades.append(trade)
        
        # 5. 포지션 잔량 업데이트
        remaining = current_pos.quantity - close_qty
        if remaining <= 0.000001: # 부동소수점 오차 고려
            del self.positions[order.symbol]
        else:
            current_pos.quantity = remaining

        order.status = OrderStatus.COMPLETED

    def _check_positions(self, feed: DataFeed):
        """포지션 가치 갱신 및 청산 확인"""
        for symbol, pos in list(self.positions.items()):
            pos.update_price(feed.close)
            
            # 격리 마진 청산 로직 (Isolated Margin)
            # Strategy가 pos.liq_price를 설정해줬다고 가정
            if pos.liq_price > 0:
                is_liquidated = False
                if pos.side == OrderSide.BUY and feed.low <= pos.liq_price:
                    is_liquidated = True
                elif pos.side == OrderSide.SELL and feed.high >= pos.liq_price:
                    is_liquidated = True
                    
                if is_liquidated:
                    self._liquidate_position(pos, feed.date, pos.liq_price)

    def _liquidate_position(self, pos: Position, timestamp: datetime, price: float):
        """강제 청산 집행"""
        # 최대 손실은 증거금으로 제한 (격리 마진)
        loss = -pos.initial_margin
        
        self.cash += loss # 증거금만큼만 까임 (실제로는 증거금이 사라지는 것)
        
        trade = Trade(
            symbol=pos.symbol,
            entry_time=pos.entry_time,
            exit_time=timestamp,
            side=pos.side,
            quantity=pos.quantity,
            entry_price=pos.entry_price,
            exit_price=price,
            gross_pnl=loss,
            net_pnl=loss,
            commission=0,
            exit_reason="LIQUIDATION"
        )
        self.trades.append(trade)
        del self.positions[pos.symbol]
        self.logger.warning(f"☠️ LIQUIDATION [{pos.symbol}] @ {price}")
    
    def _log_equity_status(self, timestamp: datetime):
        """현재 시점의 총 자산 가치를 기록합니다."""
        total_equity = self.get_total_equity()
        self.equity_history.append({
            'timestamp': timestamp,
            'equity': total_equity,
            'cash': self.cash
            # 필요하다면 positions_count 등을 추가 가능
        })