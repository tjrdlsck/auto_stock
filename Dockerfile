# 1. Python 3.10 Slim 버전을 기반으로 사용 (가볍고 안정적)
FROM python:3.13-slim

# 2. 환경 변수 설정
# 파이썬 로그가 버퍼링 없이 즉시 출력되도록 설정 (로그 확인 용이)
ENV PYTHONUNBUFFERED=1
# .pyc 파일 생성 방지
ENV PYTHONDONTWRITEBYTECODE=1

# 3. 작업 디렉토리 생성
WORKDIR /app

# 4. 시스템 의존성 설치 (numpy, hmmlearn 등 빌드에 필요)
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 5. 의존성 파일 복사 및 설치
# 캐시 효율성을 위해 requirements.txt를 먼저 복사
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 6. 전체 소스 코드 복사
COPY . .

# 7. API 서버 포트 노출 (main.py 설정값 58000)
EXPOSE 58000

# 8. 컨테이너 실행 시 메인 스크립트 실행
# src 폴더 내부의 main.py를 실행
CMD ["python", "src/main.py"]