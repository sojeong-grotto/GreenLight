import threading
import time
from unittest import case
import os
import json
import subprocess
import multiprocessing as mp
import AI_Project_Study_Color_Detect as color_detect

# os.environ["SDL_VIDEODRIVER"] = "dummy"
# os.environ["SDL_AUDIODRIVER"] = "alsa"
# os.environ["AUDIODEV"] = "plughw:UACDemoV10,0"
import pygame
from gtts import gTTS

# voice_list.json 파일에서 음성 목록 load
def load_voice_list_json(filepath: str) -> list[dict]:
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
# def load_voice_list_json(filepath: str) -> list[dict]:
#     with open(filepath, "r", encoding="utf-8") as f:
#         data = json.load(f)
#     return data

def inference_process(input_data, result_queue):
    # 무거운 AI 추론 — 별도 프로세스
    import time
    time.sleep(2)
    

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
        self._sound = {
            1: pygame.mixer.Sound(sound_list[1]),
            2: pygame.mixer.Sound(sound_list[2]),
            3: pygame.mixer.Sound(sound_list[3]),
            4: pygame.mixer.Sound(sound_list[4]),
        }

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

if __name__ == "__main__":
    tts = TTSWorker("voice_list.json")
    res_queue = mp.Queue()

    # 첫번째 parameter는 처리할 이미지 데이터(임의로 문자열로 대체)
    pc = mp.Process(target=inference_process, args=("이미지 데이터", res_queue))
    pc.start()

    # 테스트 코드
    while True:
        if tts._flag == True:
            for i in range(1, 50):
                # print(f'{i} << 번째 루프 실행')
                if i % 10 == 0:
                    tts._status = i // 10
                time.sleep(1)

    # 결과 대기 스레드 — 메인 흐름 블로킹 방지 (임시 코드, 좀 더 확인 필요)
    def wait_result():
        result = result_queue.get()
        print(f"result thread 결과: {result}")
        tts.speak(result)

    result_thread = threading.Thread(target=wait_result, daemon=True)
    result_thread.start()
    result_thread.join()

    pc.join()
    tts.stop()
