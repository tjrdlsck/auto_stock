# 코인 선물 투자 프로그램 포지션 제한 기능 분석 보고서

## 1. 개요
본 보고서는 코인 선물 투자 프로그램이 모의투자 및 실전투자 환경에서 동시에 보유할 수 있는 포지션의 개수를 제한하는 기능을 포함하고 있는지 분석한 결과를 담고 있습니다.

## 2. 분석 대상 파일
- `src/trader.py`: 핵심 매매 로직 및 포지션 관리
- `src/config.py`: 기본 설정값 정의
- `src/config.json`: 사용자 설정 파일
- `src/backtest_runner.py`: 백테스팅 로직

## 3. 분석 결과

### 3.1 포지션 제한 설정 확인
프로그램은 설정 파일(`src/config.py` 및 `src/config.json`)을 통해 최대 동시 보유 포지션 개수를 정의하고 있습니다.
- **설정 항목**: `MAX_OPEN_POSITIONS`
- **기본값**: 현재 코드상 **2**로 설정되어 있습니다. (설정 파일 내 주석에는 3으로 예시가 되어있으나 실제 값은 2)

```python
# src/config.py
"MAX_OPEN_POSITIONS": 2,
```

### 3.2 매매 로직 내 제한 기능 구현 (`src/trader.py`)
`BinanceTrader` 클래스의 `run_logic` 메서드에서 실제 매매 신호를 처리할 때 포지션 개수를 체크하는 로직이 구현되어 있습니다.

1.  **현재 포지션 수 확인**:
    매매 루프 시작 시 현재 데이터베이스(`trade_state.db`)에 저장된 포지션의 개수를 계산합니다.
    ```python
    current_pos_count = len(self.state)
    ```

2.  **진입 전 제한 확인**:
    새로운 코인에 대한 진입 신호가 발생하더라도, 현재 보유 중인 포지션 수가 설정된 최대값(`MAX_OPEN_POSITIONS`) 이상이면 진입하지 않고 다음 종목으로 넘어갑니다.
    ```python
    if current_pos_count >= self.get_conf('MAX_OPEN_POSITIONS', 3): continue
    ```
    
3.  **모드별 적용**:
    이 로직은 `run_logic` 함수 내에서 공통으로 수행되므로, **모의투자('PAPER')와 실전투자('REAL') 모드 모두에 동일하게 적용**됩니다.

### 3.3 백테스팅 로직 내 제한 기능 구현 (`src/backtest_runner.py`)
백테스팅 시에도 동일한 제한이 적용되도록 구현되어 있습니다. `RealPortfolioStrategy` 클래스 등에서 `MAX_OPEN_POSITIONS` 값을 읽어와 진입 전 확인합니다.

```python
max_pos = int(self.p.config.get('MAX_OPEN_POSITIONS', 3))
...
if open_positions >= max_pos: continue
```

## 4. 결론
분석 결과, 해당 프로그램은 **모의투자와 실전투자 모두에서 사용자가 지정한 포지션 개수(`MAX_OPEN_POSITIONS`)만큼만 주문을 실행하도록 안전장치가 구현되어 있음**을 확인하였습니다. 

따라서 설정 파일(`config.json` 또는 `config.py`)의 `MAX_OPEN_POSITIONS` 값을 변경함으로써 동시에 운영할 포지션의 수를 제어할 수 있습니다.
