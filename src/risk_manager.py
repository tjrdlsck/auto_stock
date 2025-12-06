import sys
import os
import math

# 상위 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from config.settings import Config

class RiskManager:
    """
    Risk Manager
    전략의 매매 신호를 바탕으로 안전한 진입 물량(Quantity)을 계산합니다.
    """
    
    def __init__(self, risk_per_trade=None, leverage=None):
        # 값이 들어오면 그걸 쓰고, 안 들어오면 Config 기본값 사용
        self.risk_per_trade = risk_per_trade if risk_per_trade is not None else Config.RISK_PER_TRADE
        self.leverage = leverage if leverage is not None else Config.LEVERAGE

    def calculate_entry_params(self, balance: float, entry_price: float, stop_loss: float, action: str):
        """
        자금 관리 규칙에 따른 진입 파라미터 계산 (청산가 계산 로직 추가됨)
        """
        if entry_price <= 0 or stop_loss <= 0:
            return None
    
        # 1. 리스크 금액 산정
        risk_amount = balance * self.risk_per_trade
        
        # 2. 포지션 수량 계산
        price_diff = abs(entry_price - stop_loss)
        if price_diff == 0: return None
    
        position_size = risk_amount / price_diff
        
        # 3. 레버리지 한도 체크
        notional_value = position_size * entry_price
        max_allowed_notional = balance * self.leverage
        
        if notional_value > max_allowed_notional:
            # print(f"⚠️ [RiskManager] Capped by leverage. Req: {notional_value:.2f}, Max: {max_allowed_notional:.2f}")
            position_size = max_allowed_notional / entry_price
    
        quantity = round(position_size, 4)
        
        # ---------------------------------------------------------
        # [NEW] 강제 청산가(Liquidation Price) 계산
        # 유지증거금율(Maintenance Margin) 약 0.5% 가정
        # Long Liq = Entry * (1 - 1/Lev + 0.005)
        # Short Liq = Entry * (1 + 1/Lev - 0.005)
        # ---------------------------------------------------------
        mm_rate = 0.005 # 0.5% 유지증거금
        
        if action == "BUY":
            liq_price = entry_price * (1 - (1/self.leverage) + mm_rate)
            # 청산가가 0보다 작으면 0으로 보정
            liq_price = max(0, liq_price)
        else: # SELL
            liq_price = entry_price * (1 + (1/self.leverage) - mm_rate)

        return {
            "action": action,
            "entry_price": entry_price,
            "quantity": quantity,
            "stop_loss": round(stop_loss, 2),
            "risk_amount_usdt": round(risk_amount, 2),
            "sl_distance": round(price_diff, 2),
            "liq_price": round(liq_price, 2) # [NEW] 청산가 정보 추가
        }

# 테스트 코드
if __name__ == "__main__":
    print("--- Step 3: RiskManager Verification ---")
    rm = RiskManager()
    
    # 가상 상황 1: BUY 포지션
    balance1 = 10000
    entry_price1 = 50000
    stop_loss1 = 49500 # 전략에서 결정된 손절가
    action1 = "BUY"
    
    print("\n--- Test Case 1: BUY Position ---")
    params1 = rm.calculate_entry_params(balance1, entry_price1, stop_loss1, action1)
    
    if params1:
        print("\n[>>] Calculated Trade Parameters:")
        print(f"Action: {params1['action']}")
        print(f"Entry: ${params1['entry_price']:.2f}")
        print(f"Stop Loss: ${params1['stop_loss']:.2f} (Distance: {params1['sl_distance']:.2f})")
        print(f"Quantity: {params1['quantity']} BTC")
        print(f"Risk Amount: ${params1['risk_amount_usdt']:.2f} (Loss if SL hit)")
        
        # 검증: 실제 손실액 계산
        real_loss1 = (entry_price1 - params1['stop_loss']) * params1['quantity']
        print(f"Verifying Loss: {real_loss1:.2f} (Should be close to Risk Amount)")
    else:
        print("Parameter calculation failed.")
        
    print("\n" + "="*40 + "\n")
    
    # 가상 상황 2: SELL 포지션
    balance2 = 10000
    entry_price2 = 50000
    stop_loss2 = 50500 # 전략에서 결정된 손절가
    action2 = "SELL"

    print("--- Test Case 2: SELL Position ---")
    params2 = rm.calculate_entry_params(balance2, entry_price2, stop_loss2, action2)
    
    if params2:
        print("\n[>>] Calculated Trade Parameters:")
        print(f"Action: {params2['action']}")
        print(f"Entry: ${params2['entry_price']:.2f}")
        print(f"Stop Loss: ${params2['stop_loss']:.2f} (Distance: {params2['sl_distance']:.2f})")
        print(f"Quantity: {params2['quantity']} BTC")
        print(f"Risk Amount: ${params2['risk_amount_usdt']:.2f} (Loss if SL hit)")
        
        # 검증: 실제 손실액 계산
        real_loss2 = (params2['stop_loss'] - entry_price2) * params2['quantity'] # Short의 경우 반대
        print(f"Verifying Loss: {real_loss2:.2f} (Should be close to Risk Amount)")
    else:
        print("Parameter calculation failed.")

    print("\n--- Verification Complete ---")