import threading
import time
import os
import json

import pygame
from gtts import gTTS

# voice_list.json 파일에서 음성 목록 load
def load_voice_list_json(filepath: str) -> list[dict]:
        
        # 1. 현재 실행 중인 파일(main.py 등)의 폴더 경로를 구합니다.
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 2. 파일 이름만 들어와도 현재 폴더 경로와 합쳐서 전체 경로를 만듭니다.
        # 만약 이미 전체 경로가 들어온다면 알아서 처리해줍니다.
        filepath = os.path.join(current_dir, filepath)
    
        if not os.path.exists(filepath):
            print(f"음성 목록 파일이 없습니다: {filepath}")
            return []
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

                if isinstance(data, list):
                    return data
                else:
                    print(f"잘못된 형식의 음성 목록: {filepath}")
                    return []
        except Exception as e:
            print(f"음성 목록 파일 읽기 실패: {filepath} — {e}")
            return []

class TTSWorker():
    def __init__(self, filepath):
        self.filepath = filepath
        self._temp_files = load_voice_list_json(filepath)
        self._sound = []
        self._status = None
        self._flag = False
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        # gTTS를 이용해 음성 파일 생성
        for file in self._temp_files:
            tts = gTTS(text=file['text'], lang='ko')
            tts.save(file['filename'])
            print(f"{file['filename']} 생성 완료")        

        self._flag = True

        sound_list = {i+1: item['filename'] for i, item in enumerate(self._temp_files)}
        self._sound = {i: pygame.mixer.Sound(item['filename']) for i, item in enumerate(self._temp_files, start=1)}

        while True:
            if self._status:
                filename = self._sound[self._status]

                print(f"{filename} 재생 시작")
                # pygame.mixer.music.load(filename)
                # pygame.mixer.music.play()
                self._sound[self._status].play()

                # 재생이 끝날 때까지 대기
                # while pygame.mixer.music.get_busy():
                #     time.sleep(0.1)
                print(f"{filename} 재생 완료")
                self._status = None

    def _cleanup(self):
        """atexit 또는 stop() 호출 시 남은 임시 파일 일괄 삭제"""
        for path in self._temp_files[:]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                    print(f"[TTS] 정리: {path}")
                except Exception as e:
                    print(f"[TTS] 정리 실패: {path} — {e}")
        self._temp_files.clear()

    def stop(self):
        # self._queue.put(None)
        self._thread.join()
        self._cleanup()
        pygame.mixer.quit()

pygame.mixer.pre_init(frequency=44100, size=-16, channels=2, buffer=512)
pygame.mixer.init()