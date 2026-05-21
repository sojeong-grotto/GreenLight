# Run Guide

## 설치

```bash
sudo apt update
sudo apt install -y espeak-ng mpg123 python3-opencv
pip install -r requirements.txt
```

Picamera2를 사용할 경우:

```bash
sudo apt install -y python3-picamera2
```

## 필요한 파일

```text
models/객체_int8.tflite
models/지면_int8.tflite
models/object_10_labels.txt
models/surface_7_labels.txt
```

## 카메라 실행

알림만:

```bash
python3 main.py --no-preview --fps 3
```

화면과 알림:

```bash
python3 main.py --preview --preview-fps 10 --fps 3 --surface-fps 3
```

소리 없이 확인:

```bash
python3 main.py --preview --preview-fps 10 --fps 3 --surface-fps 3 --no-speak
```

USB 카메라 번호 지정:

```bash
python3 main.py --camera 1 --preview --fps 3
```

## 동영상 확인

`images/` 폴더 영상 목록:

```bash
python3 main.py --list-videos
```

영상 이름 또는 번호로 실행:

```bash
python3 main.py --video test --preview --preview-fps 10 --fps 3 --surface-fps 3 --no-speak
python3 main.py --video 1 --preview --preview-fps 10 --fps 3 --surface-fps 3 --no-speak
```

동영상은 실제 카메라와 같은 방식으로 16:9 crop 후 512x288로 맞춘 뒤 추론합니다. preview에서는 지면 mask를 최대 `--surface-fps` 속도로 함께 표시합니다.

지면 알림은 preview에서 지면 mask가 켜져 있을 때 동작합니다. 보행 ROI 안에서 면적이 가장 넓은 지면 라벨을 기준으로 차도/이면도로/횡단보도 알림을 만들고, 점자블록은 좁게 잡혀도 우선 지면 상태로 봅니다.

Preview 키:

```text
q  종료
d  5초 앞으로 이동
a  5초 뒤로 이동
w  1분 앞으로 이동
s  1분 뒤로 이동
```

빠르게 훑기:

```bash
python3 main.py --video test --preview --fast --no-speak
```

## 화면 옵션

```bash
python3 main.py --video test --preview --window-scale 1.0 --no-speak
```

ROI 숨김:

```bash
python3 main.py --video test --preview --no-roi --no-speak
```

상태 패널과 confidence 숨김:

```bash
python3 main.py --video test --preview --no-overlay-text --no-speak
```

지면 mask와 지면 알림 끄기:

```bash
python3 main.py --video test --preview --no-surface-mask --no-speak
```
