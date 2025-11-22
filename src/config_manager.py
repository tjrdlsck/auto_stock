import os
import json
import sys

# 기존 config.py의 기본값을 가져오기 위해 임포트
# (현재 디렉토리가 src라고 가정)
try:
    from config import CONFIG as DEFAULT_CONFIG
except ImportError:
    # 경로 문제로 실패할 경우를 대비해 상위 경로 추가
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from config import CONFIG as DEFAULT_CONFIG

class ConfigManager:
    def __init__(self, file_name='config.json'):
        # config.json은 src 폴더 내부에 저장됩니다.
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.file_path = os.path.join(self.base_dir, file_name)
        self.config = {}
        
        self._load_config()

    def _load_config(self):
        """
        JSON 설정 파일이 있으면 로드하고, 없으면 기본값(config.py)을 로드하여 파일로 생성
        """
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
                print(f"✅ [Config] '{self.file_path}'에서 설정을 로드했습니다.")
            except Exception as e:
                print(f"⚠️ [Config] 설정 파일 로드 중 오류 발생: {e}. 기본값을 사용합니다.")
                self.config = DEFAULT_CONFIG.copy()
        else:
            print(f"ℹ️ [Config] 설정 파일이 없습니다. 기본값으로 '{self.file_path}'를 생성합니다.")
            self.config = DEFAULT_CONFIG.copy()
            self._save_config()

    def _save_config(self):
        """
        현재 메모리에 있는 설정을 JSON 파일로 저장
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
        설정값 가져오기
        """
        return self.config.get(key, default)

    def get_all(self):
        """
        전체 설정 딕셔너리 반환 (Web UI 표시용)
        """
        return self.config

    def set(self, key, value):
        """
        설정값 변경 및 즉시 저장
        """
        if key in self.config:
            self.config[key] = value
            saved = self._save_config()
            if saved:
                print(f"⚙️ [Config] 설정 변경됨: {key} = {value}")
            return saved
        else:
            # 새로운 키 추가도 허용할지 여부 (일단 허용)
            self.config[key] = value
            return self._save_config()

    def update_bulk(self, new_settings: dict):
        """
        여러 설정을 한 번에 업데이트 (Web UI 폼 저장용)
        """
        updated_count = 0
        for key, value in new_settings.items():
            # 기존 키가 있을 때만 업데이트하거나, 새로운 키도 허용하려면 로직 조정
            self.config[key] = value
            updated_count += 1
        
        if updated_count > 0:
            self._save_config()
            print(f"⚙️ [Config] {updated_count}개의 설정이 일괄 업데이트되었습니다.")
        return True