import threading
import time
import os
import json
import pygame
import queue
from gtts import gTTS


dicSpeechText = {"빨간 신호" : 1, "초록 신호" : 2, "점멸 신호" : 3, "위험 " : 4, "TEST" : 5}

# voice_list.json 파일에서 음성 목록 load
def load_voice_list_json(filepath: str) -> list[dict]:
        
        # 1. 현재 실행 중인 파일(main.py 등)의 폴더 경로를 구합니다.
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 2. 파일 이름만 들어와도 현재 폴더 경로와 합쳐서 전체 경로를 만듭니다.
        # 만약 이미 전체 경로가 들어온다면 알아서 처리해줍니다.
        filepath = os.path.join(current_dir, filepath)
        print(filepath)
    
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
        self.msg_queue = queue.Queue()  # 소리 문구 전용 큐

        self.filepath = filepath
        self._temp_files = load_voice_list_json(filepath)
        self._sound = {}
        self._status = None
        self._flag = False
        self._stop_event = threading.Event()
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        # gTTS를 이용해 음성 파일 생성
        for file in self._temp_files:
            try:
                tts_obj = gTTS(text=file['text'], lang='ko')
                tts_obj.save(file['filename'])
                print(f"{file['filename']} 생성 완료")
            except Exception as e:
                print(f"[TTS] gTTS 생성 실패: {e}")

        self._flag = True
        # 생성된 파일들을 pygame 사운드로 로드 (1-based index in voice file assumed)
        for i, item in enumerate(self._temp_files, start=1):
            try:
                self._sound[i] = pygame.mixer.Sound(item['filename'])
            except Exception as e:
                print(f"[TTS] sound load failed for {item.get('filename')}: {e}")

        self._ready.set()

        while not self._stop_event.is_set():
            if self._status:
                status = self._status
                self._status = None
                # enqueue and play
                self.msg_queue.put(status)
                try:
                    snd = self._sound.get(status)
                    if snd:
                        snd.play()
                        while pygame.mixer.get_busy() and not self._stop_event.is_set():
                            time.sleep(0.05)
                except Exception as e:
                    print(f"[TTS] playback error: {e}")
            else:
                time.sleep(0.05)

        # cleanup on exit
        self._cleanup()

    def _cleanup(self):
        for path in self._temp_files[:]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                    print(f"[TTS] 정리: {path}")
                except Exception as e:
                    print(f"[TTS] 정리 실패: {path} — {e}")
        self._temp_files.clear()
        try:
            pygame.mixer.quit()
        except:
            pass

    def stop(self, timeout=1.0):
        self._stop_event.set()
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            print("[TTS] thread did not stop in time")

pygame.mixer.pre_init(frequency=44100, size=-16, channels=2, buffer=512)
pygame.mixer.init()