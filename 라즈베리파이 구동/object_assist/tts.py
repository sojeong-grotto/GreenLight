import subprocess
import threading
import time
import shutil

from config import ALERT_COOLDOWN_SEC, BASE_DIR


# tts.py는 안전 이벤트 문장을 음성으로 출력합니다.
# 알림은 메인 루프를 막지 않도록 별도 thread에서 실행하고, 같은 알림 반복은 cooldown으로 제한합니다.


AUDIO_DIR = BASE_DIR / "audio"
ALERT_AUDIO_EXTS = (".wav", ".mp3")


class AlertSpeaker:
    def __init__(self, cooldown_sec=ALERT_COOLDOWN_SEC):
        self.cooldown_sec = cooldown_sec
        self.last_spoken = {}
        self.lock = threading.Lock()

    def speak(self, message, key=None, force=False):
        # key가 같으면 같은 종류의 알림으로 보고 cooldown 시간 안에는 다시 말하지 않습니다.
        key = key or message
        now = time.monotonic()
        with self.lock:
            last = self.last_spoken.get(key, 0)
            if not force and now - last < self.cooldown_sec:
                return False
            self.last_spoken[key] = now

        threading.Thread(target=self._speak_blocking, args=(message, key), daemon=True).start()
        return True

    def _speak_blocking(self, message, key):
        audio_path = self._find_alert_audio(key)
        if audio_path and self._play_audio(audio_path):
            return
        self._speak_tts(message)

    def _find_alert_audio(self, key):
        # AlertEvent.key와 같은 이름의 고정 음원이 있으면 TTS 생성 없이 바로 재생합니다.
        for ext in ALERT_AUDIO_EXTS:
            path = AUDIO_DIR / f"{key}{ext}"
            if path.exists():
                return path
        return None

    def _play_audio(self, audio_path):
        suffix = audio_path.suffix.lower()
        if suffix == ".wav":
            commands = [
                ["aplay", str(audio_path)],
                ["paplay", str(audio_path)],
                ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(audio_path)],
            ]
        elif suffix == ".mp3":
            commands = [
                ["mpg123", "-q", str(audio_path)],
                ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(audio_path)],
            ]
        else:
            commands = []

        for cmd in commands:
            if shutil.which(cmd[0]) is None:
                continue
            try:
                result = subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except FileNotFoundError:
                continue
            if result.returncode == 0:
                return True
        return False

    def _speak_tts(self, message):
        # Raspberry Pi OS에서 가장 단순한 기본값은 espeak-ng입니다.
        # Korean voice quality is limited; later replace this with Piper or a
        # Korean TTS engine if needed.
        commands = [
            ["espeak-ng", "-v", "ko", "-s", "155", message],
            ["espeak", "-v", "ko", "-s", "155", message],
        ]
        for cmd in commands:
            try:
                subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return
            except FileNotFoundError:
                continue
        print("[TTS]", message)
