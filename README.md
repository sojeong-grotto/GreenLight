# 🚶 SmartWalk — 마운트형 인도 보행 보조 장치

> AI 기반 신호등 인식 및 보행 환경 감지로 시각적 판단이 어려운 환경에서도  
> 안전한 보행을 돕는 엣지 디바이스 솔루션

---

## 📌 프로젝트 개요

**SmartWalk**는 보행자의 안전을 위해 인도에 마운트되어 실시간으로 주변 환경을 분석하는 AI 보행 보조 장치입니다.  
**Raspberry Pi 5** 위에서 동작하며, 카메라로 촬영된 영상을 AI 모델로 분석하여  
신호등 상태, 인도/차도 구분, 실내외 여부를 판별하고 상황에 맞는 안내 음성을 USB 스피커로 출력합니다.

---

## 🎯 주요 기능

### 1. 신호등 감지 및 구분
- 카메라 영상에서 신호등을 실시간으로 탐지
- 빨간불 / 초록불 / 꺼진 신호(비활성) 상태 구분
- 상태에 따른 안내 음성 자동 출력

| 신호 상태 | 안내 내용 |
|-----------|-----------|
| 🔴 빨간불 / 꺼진 신호 | "현재 빨간 신호입니다. 신호가 바뀔 때까지 기다려 주시기 바랍니다." |
| 🟢 초록불 | "초록 신호입니다. 안전하게 횡단보도를 이용하여 건너가셔도 됩니다." |
| 🟢→🔴 신호 전환 | "신호가 바뀝니다. 다음 신호를 이용해 주시기 바랍니다." |

### 2. 인도 / 차도 이탈 감지
- 보행자의 위치가 인도에서 차도로 이탈하는 경우 즉시 감지
- 경고 음성 출력으로 위험 상황 사전 차단

| 감지 상태 | 안내 내용 |
|-----------|-----------|
| 🚨 차도 이탈 감지 | "위험합니다! 차도로 나가지 마세요." |

### 3. 실내 / 실외 구분
- 촬영 영상 분석을 통해 실내/실외 환경 자동 구분
- 실외 환경에서만 신호등 감지 및 인도/차도 구분 로직 활성화

### 4. 음성 안내 시스템
- `voice_list.json`에 정의된 음성 파일 매핑 기반으로 TTS 생성
- USB 스피커를 통한 실시간 안내 음성 출력
- `pygame.mixer` + ALSA 드라이버를 활용한 지연 없는 음성 재생

---

## 🗂️ 프로젝트 구조

```
SmartWalk/
├── main.py                  # 메인 실행 파일
├── voice_list.json          # 음성 파일 매핑 정의
├── config.py                # 설정 (카메라 번호, 스피커 장치 등)
│
├── detection/
│   ├── signal_detector.py   # 신호등 감지 모듈
│   ├── zone_detector.py     # 인도/차도 구분 모듈
│   └── env_detector.py      # 실내/외 구분 모듈
│
├── audio/
│   ├── tts_worker.py        # TTS 음성 생성 스레드
│   └── player_worker.py     # pygame 음성 재생 스레드
│
├── models/
│   ├── signal_model/        # 신호등 분류 AI 모델
│   ├── zone_model/          # 인도/차도 분류 AI 모델
│   └── env_model/           # 실내/외 분류 AI 모델
│
├── training/
│   ├── collect_data.py      # 학습 데이터 수집 스크립트
│   ├── train_signal.py      # 신호등 모델 학습
│   ├── train_zone.py        # 인도/차도 모델 학습
│   └── train_env.py         # 실내/외 모델 학습
│
├── assets/
│   └── audio/               # 생성된 wav 음성 파일
│
└── requirements.txt
```

---

## ⚙️ 시스템 아키텍처

```
USB 카메라
     │
     ▼
┌───────────────────┐
│   CaptureWorker   │  (Thread) 실시간 프레임 캡처
└────────┬──────────┘
         │ frame queue
         ▼
┌───────────────────┐
│  DetectionWorker  │  (Thread) AI 모델로 상황 판별
│                   │
│  ├ 신호등 감지     │  → red / green / green_to_red / off
│  ├ 인도/차도 구분  │  → sidewalk / roadway
│  └ 실내/외 구분   │  → indoor / outdoor
└────────┬──────────┘
         │ signal code
         ▼
┌───────────────────┐
│   TTSWorker       │  (Thread) voice_list.json 기반 음성 생성
└────────┬──────────┘
         │ audio path
         ▼
┌─────────────────────────────┐
│   PlayerWorker              │  (Thread) pygame.mixer
│   ALSA → USB Speaker 출력   │
└─────────────────────────────┘

전체 동작 환경: Raspberry Pi 5
```

---

## 🧠 AI 모델 학습

인도 보행 영상 데이터를 기반으로 3가지 분류 모델을 학습합니다.

### 학습 데이터
- 직접 촬영한 인도 보행 영상 (다양한 시간대 / 날씨 / 장소)
- 신호등 상태별 라벨링 (red / green / off)
- 인도 / 차도 영역 라벨링
- 실내 / 실외 환경 라벨링

### 모델 구조
- **신호등 분류**: CNN 기반 다중 분류 모델 (OpenCV HSV 색상 분석 보조)
- **인도/차도 구분**: 시멘틱 세그멘테이션 (DeepLab / SegFormer)
- **실내/외 구분**: MobileNetV2 기반 경량 분류 모델 (Raspberry Pi 5 엣지 최적화)

---

## 🔊 음성 파일 매핑 (`voice_list.json`)

```json
[
  {"filename": "red.wav",          "text": "현재 빨간 신호입니다. 신호가 바뀔 때까지 기다려 주시기 바랍니다."},
  {"filename": "green.wav",        "text": "초록 신호입니다. 안전하게 횡단보도를 이용하여 건너가셔도 됩니다."},
  {"filename": "green_to_red.wav", "text": "신호가 바뀝니다. 다음 신호를 이용해 주시기 바랍니다."},
  {"filename": "roadway_exit.wav", "text": "위험합니다! 차도로 나가지 마세요."}
]
```

> `text` 값을 수정하면 TTS가 자동으로 새 음성 파일을 생성합니다.

---

## 🛠️ 기술 스택

### Language & Runtime
![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)

### Computer Vision & AI
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=flat-square&logo=opencv&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.x-013243?style=flat-square&logo=numpy&logoColor=white)

### Audio
![pygame](https://img.shields.io/badge/pygame-2.x-1E1E1E?style=flat-square)
![gTTS](https://img.shields.io/badge/gTTS-TTS-4285F4?style=flat-square&logo=google&logoColor=white)
![ALSA](https://img.shields.io/badge/ALSA-USB%20Speaker-FF6600?style=flat-square)

### Concurrency
![Threading](https://img.shields.io/badge/Python-Threading-green?style=flat-square)

### Hardware
![Raspberry Pi](https://img.shields.io/badge/Raspberry%20Pi-5-A22846?style=flat-square&logo=raspberrypi&logoColor=white)
![Raspberry Pi OS](https://img.shields.io/badge/Raspberry%20Pi%20OS-Bookworm%2064bit-C51A4A?style=flat-square&logo=raspberrypi&logoColor=white)
![USB Camera](https://img.shields.io/badge/USB%20Camera-OpenCV-2496ED?style=flat-square)
![USB Speaker](https://img.shields.io/badge/USB%20Speaker-ALSA-FF6600?style=flat-square)

---

## 📦 설치 및 실행

### 요구 사항
- **Raspberry Pi 5** (RAM 4GB 이상 권장)
- Raspberry Pi OS (64-bit) Bookworm
- Python 3.11 이상
- USB 카메라
- USB 스피커

### Raspberry Pi 5 환경 설정

```bash
# 시스템 업데이트
sudo apt update && sudo apt upgrade -y

# 필수 시스템 패키지
sudo apt install -y python3-pip python3-venv
sudo apt install -y libopencv-dev python3-opencv
sudo apt install -y libsdl2-dev libsdl2-mixer-dev   # pygame 의존성
sudo apt install -y alsa-utils                       # ALSA 오디오

# 가상환경 생성 (권장)
python3 -m venv venv
source venv/bin/activate
```

### USB 스피커 설정

```bash
# 연결된 오디오 장치 확인
aplay -l

# 출력 예시
# card 0: 내장사운드
# card 1: USB [USB Audio Device]  ← USB 스피커 card 번호 확인

# 음량 최대로 설정
amixer -c 1 sset 'Speaker' 100%
# Speaker 채널이 없으면
amixer -c 1 sset 'PCM' 100%

# 설정 영구 저장 (재부팅 후에도 유지)
sudo alsactl store
```

### 프로젝트 설치

```bash
git clone https://github.com/sojeong-grotto/GreenLight.git
cd GreenLight
pip install -r requirements.txt
```

### 설정 (`config.py`)

```python
import os

# USB 스피커 설정 (import pygame 이전에 반드시 선언)
os.environ["SDL_VIDEODRIVER"] = "dummy"       # 디스플레이 없는 환경
os.environ["SDL_AUDIODRIVER"] = "alsa"        # ALSA 드라이버
os.environ["AUDIODEV"]        = "plughw:1,0"  # USB 스피커 (aplay -l 확인)

# 카메라
CAMERA_INDEX = 0       # USB 카메라 번호

# 오디오
SAMPLE_RATE  = 44100
CHANNELS     = 2
BUFFER_SIZE  = 512
```

### 실행

```bash
python main.py
```

---

## 📋 요구 패키지 (`requirements.txt`)

```
opencv-python==4.9.0.80
numpy==1.26.4
pygame==2.5.2
gTTS==2.5.1
torch==2.2.0
torchvision==0.17.0
```

---

## 🔄 신호 처리 흐름

```
1. USB 카메라 프레임 캡처 (CaptureWorker)
        ↓
2. 실내/외 판별 → 실내면 처리 중단
        ↓
3. 신호등 감지 → red / green / green_to_red / off
        ↓
4. 인도/차도 구분 → roadway 이탈 감지 시 즉시 경고
        ↓
5. 상황 코드 → TTSWorker → PlayerWorker → USB 스피커 출력
```

---

## 📸 시연 영상 / 사진

> 추후 추가 예정

---

## 🙋 개발자

| 이름 | 역할 | GitHub |
|------|------|--------|
| 김소정 |  | [@sojeong-grotto](https://github.com/sojeong-grotto) |
