# ObjectAssist Runtime

ObjectAssist는 라즈베리파이5에서 객체 감지 모델 하나만 사용해 보행 중 가까운 사람, 차량, 고정 장애물을 알려주는 경량 보행 보조 런타임입니다.

이 프로젝트는 처음에는 지면 분류와 신호등 인식까지 포함하려 했지만, 현재 모델 성능을 기준으로 안정적으로 사용할 수 있는 객체 인식 중심으로 재구성했습니다. 지면 모델, 신호등 알림, 벤치마크 스크립트, 모델 프로브, 스무딩 실험 코드는 제거했고, 실제 카메라 실행과 동영상 검증에 필요한 코드만 남겼습니다.

## 목표

- 카메라 영상에서 사람, 차량, 고정 장애물을 감지합니다.
- 객체의 위치, 크기 변화, ROI 진입 여부를 이용해 위험도를 판단합니다.
- 같은 알림이 짧은 시간 안에 반복되지 않도록 제어합니다.
- 라즈베리파이5에서 가능한 한 가볍게 실행되도록 bbox 기반 후처리만 사용합니다.
- 동영상 파일을 실제 카메라 입력과 같은 방식으로 처리해 알림 기준을 조정할 수 있게 합니다.

## 폴더 구조

```text
object_assist/
├─ main.py              # 카메라/동영상 실행 루프, preview, 터미널/음성 알림
├─ camera.py            # Picamera2 또는 OpenCV 카메라 입력, 16:9 crop/resize
├─ detector.py          # TFLite 모델 실행, 모델 출력을 Detection으로 변환
├─ safety.py            # 사람/차량/고정 장애물 알림 알고리즘
├─ tts.py               # 고정 음원 또는 TTS 음성 알림
├─ config.py            # 모델 경로, confidence, ROI, 알림 기준값
├─ requirements.txt     # Python 의존성
├─ RUN_GUIDE.md         # 실행 명령 중심 안내
├─ models/
│  ├─ 객체_int8.tflite
│  ├─ object_10_labels.txt
│  ├─ 지면_int8.tflite
│  └─ surface_7_labels.txt
├─ audio/
│  ├─ moving_obstacle.mp3
│  └─ static_obstacle.mp3
└─ images/
   └─ test.mp4
```

## 전체 처리 흐름

```text
카메라 또는 동영상 프레임
        ↓
16:9 crop + 512x288 resize
        ↓
512x512 letterbox 모델 입력 생성
        ↓
TFLite 객체 모델 추론
        ↓
Detection 목록 생성
        ↓
ROI와 bbox 변화 기반 안전 판단
        ↓
터미널 로그, preview 오버레이, 음성 알림
```

## 영상 처리 구조

`camera.py`는 입력 프레임을 항상 `512x288` 크기의 16:9 영상으로 맞춥니다. 카메라가 다른 비율의 영상을 주면 먼저 중앙 crop으로 16:9를 만든 뒤 resize합니다. 이렇게 하면 객체가 좌우 또는 상하로 찌그러지지 않고, ROI 좌표도 항상 같은 기준으로 동작합니다.

모델은 `512x512` 입력을 사용합니다. 따라서 `detector.py`는 `512x288` 프레임을 바로 늘리지 않고, 비율을 유지한 채 `512x512` 캔버스에 넣는 letterbox 방식을 사용합니다. 남는 영역은 `LETTERBOX_COLOR = 114`로 채웁니다.

추론 후 모델 bbox 좌표는 다시 letterbox padding을 제거하고 원래 `512x288` 카메라 프레임 기준의 정규화 좌표로 복원됩니다. 이후 `safety.py`와 preview 오버레이는 모두 이 정규화 좌표를 사용합니다.

## 모델 구조

사용 모델:

```text
models/객체_int8.tflite
models/지면_int8.tflite
```

라벨 파일:

```text
models/object_10_labels.txt
models/surface_7_labels.txt
```

현재 코드는 NMS가 포함된 Ultralytics 계열 TFLite 출력을 기대합니다. 각 detection row는 다음 형태를 기준으로 해석합니다.

```text
x1, y1, x2, y2, score, class_id, ...
```

객체 모델은 bbox, confidence, class id만 사용합니다. 지면 모델은 preview가 켜져 있을 때만 segmentation mask를 복원해 화면에 얹습니다. 지면 mask 복원은 연산량이 있으므로 기본 최대 3 FPS로 제한합니다.

Detection은 다음 정보로 정리됩니다.

```python
class_id
label
score
box  # 512x288 프레임 기준 정규화 좌표 x1,y1,x2,y2
mask # 지면 preview용 HxW mask, 객체 알림 로직에는 사용하지 않음
```

## TFLite 실행 방식

`detector.py`는 실행 가능한 TFLite interpreter를 순서대로 시도합니다.

1. `ai_edge_litert`
2. `tflite_runtime`
3. `tensorflow.lite.Interpreter`

기본 설정은 `config.py`에 있습니다.

```python
TFLITE_BACKEND = "tensorflow:default"
TFLITE_NUM_THREADS = 1
TFLITE_DISABLE_XNNPACK = True
```

라즈베리파이5에서 테스트한 결과 현재 환경에서는 1 thread가 가장 안정적이어서 기본값으로 둡니다. 다른 환경에서 바꾸려면 실행 전에 환경변수로 지정할 수 있습니다.

```bash
TFLITE_NUM_THREADS=2 python3 main.py --preview --fps 3
```

## Confidence 기준

전체 라벨 confidence 기준은 `config.py`의 한 값으로 조정합니다.

```python
DEFAULT_CLASS_CONF_THRESHOLD = 0.30
```

모든 라벨은 기본적으로 이 값을 따릅니다.

```python
CLASS_CONF_THRESHOLDS = {
    "person": DEFAULT_CLASS_CONF_THRESHOLD,
    "vehicle": DEFAULT_CLASS_CONF_THRESHOLD,
    ...
}
```

특정 라벨만 다르게 보고 싶으면 해당 라벨만 숫자로 바꾸면 됩니다.

```python
"vehicle": 0.35,
"person": DEFAULT_CLASS_CONF_THRESHOLD,
```

## ROI 구조

ObjectAssist는 두 개의 ROI를 사용합니다.

### 보행 경로 ROI

```python
PATH_ROI_POLYGON = [
    (0.44, 0.88),
    (0.56, 0.88),
    (0.64, 0.98),
    (0.36, 0.98),
]
```

보행 경로 ROI는 사용자가 실제로 걸어갈 가능성이 높은 하단 중앙 영역입니다. preview에서는 빨간색 사다리꼴로 표시됩니다. 고정 장애물과 사람 판단에 사용합니다.

### 차량 ROI

```python
VEHICLE_ROI_POLYGON = [
    (0.41, 0.70),
    (0.59, 0.70),
    (0.70, 0.98),
    (0.30, 0.98),
]
```

차량 ROI는 보행 경로보다 넓습니다. 차량은 옆에서 들어오거나 빠르게 접근하는 경우가 있어 보행 경로 ROI보다 먼저 감지해야 하기 때문입니다. preview에서는 주황색 사다리꼴로 표시됩니다.

차량 ROI 판정에는 bbox 하단변과 ROI가 겹치는 비율을 사용합니다. 차량의 상단이나 좌우 변 중심까지 사용하면 큰 bbox가 ROI 바깥에 있어도 일부 기준점이 사다리꼴에 걸려 이동 알림이 과하게 뜰 수 있기 때문입니다.

```text
bbox 하단변을 여러 샘플 점으로 나눔
샘플 중 ROI 안에 들어간 비율을 계산
기본 10% 이상이면 ROI에 걸친 것으로 판단
```

현재 기준은 `ROI_BOTTOM_OVERLAP_RATIO = 0.10`입니다. 점 하나가 우연히 걸치는 방식보다 안정적이고, 하단 중심만 보는 방식보다 유연합니다.

## 알림 종류

ObjectAssist의 주요 알림은 객체 알림과 지면 알림으로 나뉩니다.

```text
moving_obstacle
static_obstacle
person_near
roadway_entry
alley_entry
crosswalk_detected
braille_blocks
```

알림은 priority가 높은 순서로 정렬됩니다. 한 추론 프레임에서 실제로 출력하는 알림 수는 기본 1개입니다.

```python
ALERTS_PER_INFERENCE = 1
```

같은 객체에 대한 같은 알림은 cooldown 시간 안에 다시 울리지 않습니다.

```python
ALERT_COOLDOWN_SEC = 4.0
```

객체 id는 추적 id 또는 위치 grid 기반 signature를 사용합니다. 그래서 같은 종류의 알림이라도 새로운 객체라면 별도로 울릴 수 있습니다.

## 고정 장애물 알고리즘

고정 장애물 라벨:

```python
STATIC_OBSTACLE_LABELS = {
    "vertical_obstacle",
    "temporary_obstacle",
    "bench",
    "traffic_sign",
    "bus_taxi_stop",
}
```

고정 장애물은 다음 조건을 만족해야 합니다.

```text
1. bbox 하단변의 10% 이상이 PATH_ROI_POLYGON과 겹침
2. bbox 면적 비율이 STATIC_NEAR_AREA_RATIO 이상
3. STATIC_CONFIRM_FRAMES 프레임 이상 연속 확인됨
4. 같은 프레임의 vehicle bbox와 크게 겹치지 않음
```

현재 기준:

```python
STATIC_NEAR_AREA_RATIO = 0.0015
STATIC_CONFIRM_FRAMES = 3
ROI_BOTTOM_OVERLAP_RATIO = 0.10
STATIC_VEHICLE_SUPPRESS_OVERLAP_RATIO = 0.35
```

고정 장애물은 bbox 하단변이 보행 경로 ROI와 일정 비율 이상 겹쳐야 후보가 됩니다. 중심점은 사용하지 않습니다. 기둥이나 표지판처럼 bbox가 길게 잡힌 객체가 실제 보행 경로 밖에 있는데 중심만 ROI에 걸려 알림이 뜨는 상황을 줄이기 위해서입니다.
차량이 고정체 라벨로도 중복 검출되는 경우를 줄이기 위해, 고정체 bbox 중심이 vehicle bbox 안에 있거나 고정체 bbox 면적의 35% 이상이 vehicle bbox와 겹치면 고정체 후보에서 제외합니다.
차량은 이동 판단을 위해 2번 관측을 유지하지만, 고정 장애물은 안정성을 위해 3번 연속 확인된 뒤 알림을 냅니다.

알림:

```text
앞에 장애물이 있습니다. 주의하세요.
```

## 사람 알림 알고리즘

사람 관련 라벨:

```python
PERSON_LABELS = {"person", "mobility_aid"}
```

사람 알림은 다음 조건을 봅니다.

```text
1. bbox 하단 anchor가 PATH_ROI_POLYGON 안에 있음
2. bbox 중심 x가 FORWARD_ZONE_X1 ~ FORWARD_ZONE_X2 안에 있음
3. bbox 면적이 충분히 크거나, bbox 하단이 충분히 아래에 있거나, bbox 아래변 길이가 충분히 넓음
```

현재 기준:

```python
FORWARD_ZONE_X1 = 0.35
FORWARD_ZONE_X2 = 0.65
PERSON_NEAR_AREA_RATIO = 0.16
PERSON_NEAR_BOTTOM_Y = 0.82
PERSON_NEAR_WIDTH_RATIO = 0.18
```

하단 anchor는 bbox 하단의 25%, 50%, 75% 지점입니다. 사람 bbox는 다리와 몸통이 길게 잡히기 때문에 하단 지점을 보는 편이 보행 충돌 가능성을 더 잘 반영합니다.

알림:

```text
앞에 사람이 가깝습니다. 천천히 이동하세요.
```

## 차량 알림 알고리즘

차량 라벨:

```python
MOVING_RISK_LABELS = {"vehicle"}
```

차량은 먼저 간단한 중심점 기반 tracking을 합니다. 이전 프레임의 차량 중심과 현재 차량 중심의 거리가 `MOVING_MATCH_DISTANCE`보다 가까우면 같은 차량 track으로 봅니다.

현재 기준:

```python
MOVING_MATCH_DISTANCE = 0.20
MOVING_TRACK_MAX_GAP_SEC = 1.2
MOVING_APPROACH_WINDOW_SEC = 1.5
MOVING_MIN_OBSERVATIONS = 2
```

차량 track은 최근 `1.5초` 이내 history만 유지합니다. 이 history에서 bbox 면적 변화, 하단 이동, 중심 이동을 계산합니다.

### 이동 차량

이동 차량 알림은 정면 접근과 측면 진입 중 하나를 만족하면 발생합니다.

정면 접근 조건:

```text
1. 차량 기준점이 VEHICLE_ROI_POLYGON 안에 있음
2. bbox 면적 >= MOVING_NEAR_AREA_RATIO
3. bbox 면적이 처음 관측보다 MOVING_APPROACH_SCALE배 이상 커짐
4. 면적 증가 속도 >= MOVING_FAST_AREA_GROWTH_PER_SEC
5. bbox 하단 y가 MOVING_FRONTAL_MIN_BOTTOM_SHIFT 이상 아래로 이동
```

현재 기준:

```python
MOVING_NEAR_AREA_RATIO = 0.05
MOVING_APPROACH_SCALE = 1.18
MOVING_FAST_AREA_GROWTH_PER_SEC = 0.45
MOVING_FRONTAL_MIN_BOTTOM_SHIFT = 0.02
```

이 조건은 정면에서 빠르게 다가오는 차량이나 오토바이를 잡기 위한 기준입니다. ROI가 이미 위험 영역을 제한하므로 x축 이동 조건은 제거했습니다.

측면 진입 조건:

```text
1. 차량 기준점이 VEHICLE_ROI_POLYGON 안에 있음
2. bbox 중심이 화면 중앙 쪽으로 이동함
3. x축 이동량 >= MOVING_LATERAL_MIN_X_SHIFT
4. bbox 면적 >= MOVING_LATERAL_MIN_AREA_RATIO
5. bbox 하단 y >= MOVING_LATERAL_MIN_BOTTOM_Y
```

현재 기준:

```python
MOVING_LATERAL_MIN_X_SHIFT = 0.04
MOVING_LATERAL_MIN_AREA_RATIO = 0.025
MOVING_LATERAL_MIN_BOTTOM_Y = 0.74
```

이 조건은 옆에서 들어오는 차량을 조금 더 빨리 잡기 위한 기준입니다.

알림:

```text
차량이 가까워지고 있습니다. 주의하세요.
```

### 멈춰있는 차량

현재 안정 버전에서는 멈춰있는 차량을 고정 장애물로 알리지 않습니다. 같은 차량이 이동체와 멈춘 차량 사이를 오가며 분류되면 알림이 흔들릴 수 있기 때문입니다. `vehicle` 라벨은 정면 접근 또는 측면 진입 조건을 만족할 때만 `moving_obstacle` 알림을 만듭니다.

## 지면 알림 알고리즘

지면 알림은 preview에서 지면 mask가 켜져 있을 때 동작합니다. 보행 경로 ROI 안에서 면적이 가장 넓은 지면 라벨을 현재 경로 상태로 보고, 같은 후보가 연속 확인되면 알림을 만듭니다. 단, 점자블록은 면적이 좁아도 보행 의미가 크기 때문에 우선 라벨로 봅니다.

현재 기준:

```python
SURFACE_INFERENCE_FPS = 3.0
SURFACE_CONFIRM_FRAMES = 4
SURFACE_MIN_DOMINANT_RATIO = 0.08
SURFACE_BRAILLE_PRIORITY_RATIO = 0.03
```

알림:

```text
roadway_entry      전방이 차도입니다. 주의하세요.
alley_entry        전방이 이면도로입니다. 주의하세요.
crosswalk_detected 전방에 횡단보도가 있습니다.
braille_blocks     전방에 점자블록이 있습니다.
```

차도/이면도로는 위험 알림으로 우선순위를 높게 두고, 횡단보도/점자블록은 안내 알림으로 둡니다.

## Preview 오버레이

Preview에는 다음 정보가 표시됩니다.

- 빨간색 보행 ROI
- 주황색 차량 ROI
- 감지 bbox
- bbox confidence 숫자
- 현재 추론 FPS
- 지연 시간
- 현재 알림 상태
- 최근 알림 로그

성능을 위해 기본 preview 확대 배율은 `1.0`입니다. 즉 내부 처리 프레임인 `512x288` 그대로 표시합니다.

```bash
python3 main.py --video test --preview --window-scale 1.0 --no-speak
```

상태 패널과 confidence 숫자가 필요 없으면 끌 수 있습니다.

```bash
python3 main.py --video test --preview --no-overlay-text --no-speak
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

동영상 목록:

```bash
python3 main.py --list-videos
```

동영상 실행:

```bash
python3 main.py --video test --preview --preview-fps 10 --fps 3 --surface-fps 3 --no-speak
```

preview에서는 객체 bbox와 함께 지면 segmentation mask가 표시됩니다. 지면 mask와 지면 알림이 필요 없으면 `--no-surface-mask`를 추가합니다.

동영상은 카메라와 같은 전처리 경로를 사용합니다. 따라서 동영상으로 확인한 ROI와 알림 기준은 실제 카메라 실행과 거의 같은 기준으로 해석할 수 있습니다.

동영상 preview 키:

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

## 음성 알림

`tts.py`는 먼저 `audio/` 폴더에서 알림 key와 같은 이름의 고정 음원을 찾습니다.

```text
audio/moving_obstacle.mp3
audio/static_obstacle.mp3
```

고정 음원이 있으면 `mpg123` 또는 `ffplay`로 재생합니다. 해당 파일이 없거나 재생 명령을 사용할 수 없으면 `espeak-ng` 또는 `espeak`로 문장을 읽습니다.

음성을 끄려면:

```bash
python3 main.py --preview --no-speak
```

## 성능 관련 정리

현재 런타임은 성능을 위해 다음 항목을 제한하거나 제거했습니다.

- 지면 segmentation mask 복원은 preview에서만 사용하고 기본 3 FPS로 제한
- 신호등 알림
- bbox smoothing 실험 코드
- benchmark/probe 스크립트
- raw output debug summary

동영상 preview에서는 별도 worker가 가장 최신 프레임만 추론합니다. 추론이 늦어져도 오래된 프레임이 큐에 쌓이지 않도록 pending frame을 하나만 유지합니다.

## 한계

- 단안 카메라만 사용하므로 실제 거리와 속도를 직접 측정하지 않습니다.
- bbox 면적 변화는 카메라 사용자 본인의 이동에도 영향을 받습니다.
- 정면에서 다가오는 차량과 사용자가 멈춘 차량 쪽으로 걸어가는 상황은 완전히 구분하기 어렵습니다.
- 알림 기준은 실제 장착 위치, 카메라 화각, 보행 속도에 따라 조정이 필요합니다.

이 프로젝트는 “확실한 위험만 완벽히 판정”하기보다는 라즈베리파이5에서 실시간에 가깝게 돌릴 수 있는 단순하고 조정 가능한 보행 보조 기준을 목표로 합니다.
