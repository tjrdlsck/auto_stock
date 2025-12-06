import logging
import os
from config.settings import Config

def setup_logger(name="AI_Trader"):
    """
    로거 설정:
    1. 화면(StreamHandler): 설정된 레벨(INFO/WARNING)에 따라 출력
    2. 파일(FileHandler): 무조건 모든 정보(DEBUG)를 기록
    """
    # 로그 저장 폴더 생성
    if not os.path.exists(Config.SAVE_DIR):
        os.makedirs(Config.SAVE_DIR)
        
    log_file_path = os.path.join(Config.SAVE_DIR, "system.log")

    # 로거 객체 생성
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG) # 로거 자체는 모든 정보를 받음

    # 중복 출력 방지 (이미 핸들러가 있으면 추가 안 함)
    if logger.hasHandlers():
        return logger

    # 1. 파일 핸들러 (모든 기록을 남김 - 디버깅용)
    file_handler = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG) # 파일엔 모든 걸 적는다
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    # 2. 콘솔 핸들러 (화면 출력용 - 속도용)
    console_handler = logging.StreamHandler()
    
    # settings.py의 LOG_LEVEL에 따라 화면 출력 양 조절
    # 최적화할 땐 'WARNING', 평소엔 'INFO'로 설정하면 됨
    log_level = getattr(logging, Config.LOG_LEVEL.upper(), logging.INFO)
    console_handler.setLevel(log_level)
    
    console_formatter = logging.Formatter('%(message)s') # 화면엔 깔끔하게 메시지만
    console_handler.setFormatter(console_formatter)

    # 핸들러 등록
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger