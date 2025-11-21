# 1. 파이썬 3.13 슬림 버전
FROM python:3.13-slim

# 2. 타임존 설정 (한국 시간)
ENV TZ=Asia/Seoul
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# 3. 필수 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 4. 도커 내 기본 작업 폴더 설정
WORKDIR /app

# 5. 의존성 파일 복사 및 설치
# (requirements.txt가 src 바깥에 있으므로 /app에 복사)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. 소스 코드 전체 복사
# (현재 폴더의 모든 것을 /app으로 복사 -> /app/src가 생김)
COPY . .

# [핵심 수정] 7. 실행 작업 경로를 src 내부로 변경
# 파이썬 코드가 src 안에 있으므로 여기로 들어가야 import 에러가 안 납니다.
WORKDIR /app/src

# 8. 실행 명령어
CMD ["python", "discord_main.py"]