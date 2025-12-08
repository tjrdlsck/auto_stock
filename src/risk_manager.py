import sys
import os
import math

# 상위 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from config.settings import Config

class RiskManager:
    """
    [Step 5 Modified]
    Risk Manager with Safety Guard
    - 'Wide SL' 전략 시 발생할 수 있는 '손절가 < 청산가' 역전 현상을 방지합니다.
    - 청산가 도달 직전(0.2% 버퍼)에 강제로 손절 나가도록 파라미터를 보정합니다.
    """
    
    def __init__(self, risk_per_trade=None, leverage=None):
        self.risk_per_trade = risk_per_trade if risk_per_trade is not None else Config.RISK_PER_TRADE
        self.leverage = leverage if leverage is not None else Config.LEVERAGE

    def calculate_entry_params(self, balance: float, entry_price: float, stop_loss: float, action: str):
        """
        진입 파라미터 계산 및 안전장치 가동
        """
        if entry_price <= 0 or stop_loss <= 0:
            return None
    
        # 1. 기초 리스크 금액 산정
        risk_amount = balance * self.risk_per_trade
        
        # 2. 포지션 수량 계산 (기존 손절가 기준)
        price_diff = abs(entry_price - stop_loss)
        if price_diff == 0: return None
    
        position_size = risk_amount / price_diff
        
        # 3. 레버리지 한도 체크 (Notional Value Cap)
        notional_value = position_size * entry_price
        max_allowed_notional = balance * self.leverage
        
        if notional_value > max_allowed_notional:
            # 레버리지 한도 초과 시 수량 삭감
            position_size = max_allowed_notional / entry_price
    
        quantity = round(position_size, 4)
        if quantity <= 0: return None
        
        # ---------------------------------------------------------
        # [Safety Guard] 청산가 계산 및 손절가 보정
        # ---------------------------------------------------------
        # 유지증거금율 (Maintenance Margin Rate) - 바이낸스 기준 약 0.5% 가정
        mm_rate = 0.005 
        
        # 청산가와의 안전거리 버퍼 (Slippage 고려 0.2%)
        safety_buffer = 0.002 

        if action == "BUY":
            # Long Liquidation Formula (Isolated)
            # Liq = Entry * (1 - 1/Lev + mm_rate)
            liq_price = entry_price * (1 - (1/self.leverage) + mm_rate)
            liq_price = max(0, liq_price)
            
            # [핵심] Safety Guard
            # 만약 전략이 정한 손절가(Wide SL)가 청산가보다 낮거나 너무 가깝다면?
            safe_threshold = liq_price * (1 + safety_buffer)
            
            if stop_loss <= safe_threshold:
                # print(f"⚠️ [RiskManager] SL too wide! Adjusting: {stop_loss} -> {safe_threshold:.2f} (Liq: {liq_price:.2f})")
                stop_loss = safe_threshold

        else: # SELL
            # Short Liquidation Formula
            # Liq = Entry * (1 + 1/Lev - mm_rate)
            liq_price = entry_price * (1 + (1/self.leverage) - mm_rate)
            
            # Safety Guard for Short
            safe_threshold = liq_price * (1 - safety_buffer)
            
            if stop_loss >= safe_threshold:
                stop_loss = safe_threshold

        # 최종 리스크 재계산 (보정된 SL 기준 실제 손실액)
        # 수량을 줄이진 않고, 손절을 빨리 치는 방식으로 대응 (손실액 감소 효과)
        final_price_diff = abs(entry_price - stop_loss)
        actual_risk_amount = final_price_diff * quantity

        return {
            "action": action,
            "entry_price": entry_price,
            "quantity": quantity,
            "stop_loss": round(stop_loss, 2),
            "risk_amount_usdt": round(actual_risk_amount, 2),
            "liq_price": round(liq_price, 2)
        }

if __name__ == "__main__":
    # 간단 테스트
    rm = RiskManager(leverage=5.0) # 5배 레버리지
    
    print("--- Safety Guard Test (Long 5x) ---")
    entry = 50000
    # 의도적으로 아주 먼 손절가 설정 (청산가보다 아래)
    # 5배 레버리지면 청산가가 약 40,000불 근처임. SL을 38,000으로 설정해봄.
    dangerous_sl = 38000 
    
    params = rm.calculate_entry_params(10000, entry, dangerous_sl, "BUY")
    
    if params:
        print(f"Entry: ${params['entry_price']}")
        print(f"Original SL: ${dangerous_sl}")
        print(f"Liquidation Price: ${params['liq_price']}")
        print(f"Adjusted Safe SL: ${params['stop_loss']} (Should be > Liq Price)")
    else:
        print("Calculation Failed")