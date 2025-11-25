# 1. Python 3.13 Slim 버전을 기반으로 사용 (안정성 및 호환성 확보)
FROM python:3.13-slim

# 2. 환경 변수 설정
# 파이썬 출력 버퍼링 비활성화 (로그 즉시 출력)
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# 타임존 설정 (한국 시간)
ENV TZ=Asia/Seoul


# 3. 작업 디렉토리 생성
WORKDIR /app

# 4. 시스템 의존성 설치 (numpy, hmmlearn 빌드용)
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 5. 의존성 파일 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 6. 전체 소스 코드 복사
COPY . .

# 7. 데이터 및 모델 디렉토리 생성 (권한 문제 방지용)
RUN mkdir -p /app/data /app/models

# 7. API 서버 포트 노출 (main.py 설정값 58000)
EXPOSE 58000

# 8. 컨테이너 실행
CMD ["python", "src/main.py"]