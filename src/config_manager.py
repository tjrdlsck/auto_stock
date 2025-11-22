import os
import json
import sys
import copy

# ---------------------------------------------------------
# [설정 로드 경로 처리]
# ---------------------------------------------------------
# 현재 파일의 디렉토리
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 상위 디렉토리 등을 sys.path에 추가하여 config.py를 찾을 수 있게 함
sys.path.append(BASE_DIR)

# 기본 설정(DEFAULT_CONFIG)을 config.py에서 가져옴
try:
    from config import CONFIG as DEFAULT_CONFIG
except ImportError:
    # 만약 config.py를 못 찾으면 빈 딕셔너리로 처리 (또는 에러 로그)
    print("⚠️ [ConfigManager] config.py를 찾을 수 없어 기본값이 비어있습니다.")
    DEFAULT_CONFIG = {}

class ConfigManager:
    def __init__(self, file_name='config.json'):
        """
        설정 관리자 초기화
        :param file_name: 설정을 저장할 JSON 파일명 (기본: config.json)
        """
        self.file_path = os.path.join(BASE_DIR, file_name)
        self.config = {}
        
        # 초기화 시 설정 로드
        self._load_config()

    def _load_config(self):
        """
        JSON 파일이 존재하면 로드하고, 없거나 오류가 발생하면 기본값으로 초기화
        """
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
                print(f"✅ [Config] '{self.file_path}'에서 설정을 로드했습니다.")
            except Exception as e:
                print(f"⚠️ [Config] 파일 로드 오류({e}). 기본값으로 복원합니다.")
                self.reset_to_defaults()
        else:
            print(f"ℹ️ [Config] 설정 파일이 없습니다. 기본값으로 생성합니다.")
            self.reset_to_defaults()

    def _save_config(self):
        """
        현재 메모리 상의 설정을 JSON 파일로 저장
        """
        try:
            with open(self.file_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=4, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"❌ [Config] 설정 저장 실패: {e}")
            return False

    def get(self, key, default=None):
        """
        특정 설정값 조회
        """
        return self.config.get(key, default)

    def get_all(self):
        """
        전체 설정 딕셔너리 반환 (API 응답용)
        """
        return self.config

    def set(self, key, value):
        """
        단일 설정값 변경 및 저장
        """
        self.config[key] = value
        saved = self._save_config()
        if saved:
            print(f"⚙️ [Config] 설정 변경됨: {key} = {value}")
        return saved

    def update_bulk(self, new_settings: dict):
        """
        [수정됨] 여러 설정을 한 번에 업데이트 (Web UI 폼 저장용)
        - 입력된 값의 타입을 기존 설정의 타입에 맞춰 변환 시도
        - 예: "BTC/USDT, ETH/USDT" (String) -> ["BTC/USDT", "ETH/USDT"] (List)
        """
        updated_count = 0
        
        for key, value in new_settings.items():
            # 1. 기존 키가 존재하는 경우, 타입 안전성 확인
            if key in self.config:
                original_value = self.config[key]
                original_type = type(original_value)
                
                try:
                    # 리스트 타입인데 입력이 문자열로 온 경우 (CSV 형태)
                    if original_type == list and isinstance(value, str):
                        # 콤마로 분리하고 공백 제거
                        parsed_list = [item.strip() for item in value.split(',') if item.strip()]
                        self.config[key] = parsed_list
                    
                    # 불리언 타입인데 입력이 문자열로 온 경우
                    elif original_type == bool and isinstance(value, str):
                        self.config[key] = (value.lower() == 'true')
                    
                    # 숫자(int/float) 타입 처리
                    elif original_type == int:
                        self.config[key] = int(value)
                    elif original_type == float:
                        self.config[key] = float(value)
                        
                    # 그 외에는 그대로 대입
                    else:
                        self.config[key] = value
                        
                except Exception as e:
                    print(f"⚠️ [Config] '{key}' 값 변환 실패 ({value}): {e}. 변경하지 않습니다.")
                    continue
            
            # 2. 새로운 키인 경우 그냥 추가
            else:
                self.config[key] = value
            
            updated_count += 1
        
        if updated_count > 0:
            self._save_config()
            print(f"⚙️ [Config] {updated_count}개의 설정이 일괄 업데이트되었습니다.")
            return True
        return False

    def reset_to_defaults(self):
        """
        [추가됨] 현재 설정을 config.py에 정의된 기본값으로 초기화하고 저장
        """
        # deepcopy를 사용하여 원본 참조가 꼬이지 않게 함
        self.config = copy.deepcopy(DEFAULT_CONFIG)
        self._save_config()
        print("♻️ [Config] 모든 설정이 기본값으로 초기화되었습니다.")
        return self.config