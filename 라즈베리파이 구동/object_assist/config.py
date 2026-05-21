import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"


# 객체 모델과 지면 모델 런타임에서 사용할 모델/라벨 파일입니다.
OBJECT_MODEL_PATH = MODEL_DIR / "객체_int8.tflite"
OBJECT_LABEL_PATH = MODEL_DIR / "object_10_labels.txt"
SURFACE_MODEL_PATH = MODEL_DIR / "지면_int8.tflite"
SURFACE_LABEL_PATH = MODEL_DIR / "surface_7_labels.txt"


# 카메라/동영상 프레임은 512x288 16:9로 맞춥니다.
# 모델 입력은 512x512이므로 detector.py에서 letterbox로 위아래 padding을 붙입니다.
CAMERA_WIDTH = 512
CAMERA_HEIGHT = 288
CAMERA_FPS = 15
CAMERA_CROP_TO_TARGET_ASPECT = True

MODEL_INPUT_WIDTH = 512
MODEL_INPUT_HEIGHT = 512
LETTERBOX_COLOR = 114


# 런타임 기본값입니다.
# TFLITE_NUM_THREADS는 라즈베리파이5 실측에서 1이 가장 안정적이어서 기본값으로 둡니다.
SHOW_PREVIEW = False
USE_MOCK_DETECTOR_IF_MODEL_MISSING = False
TFLITE_NUM_THREADS = int(os.environ.get("TFLITE_NUM_THREADS", "1"))
TFLITE_DISABLE_XNNPACK = True
TFLITE_BACKEND = "tensorflow:default"
OBJECT_INFERENCE_FPS = 8.0
SURFACE_INFERENCE_FPS = 3.0
PREVIEW_FPS = 10.0


# 감지 confidence 기준입니다.
# 전체 라벨 기준을 한 번에 바꾸려면 DEFAULT_CLASS_CONF_THRESHOLD만 바꾸면 됩니다.
# 특정 라벨만 다르게 보고 싶으면 CLASS_CONF_THRESHOLDS에서 해당 라벨만 숫자로 덮어쓰면 됩니다.
DEFAULT_CLASS_CONF_THRESHOLD = 0.30
DETECTION_CONF_THRESHOLD = DEFAULT_CLASS_CONF_THRESHOLD
CLASS_CONF_THRESHOLDS = {
    "person": DEFAULT_CLASS_CONF_THRESHOLD,
    "vehicle": DEFAULT_CLASS_CONF_THRESHOLD,
    "mobility_aid": DEFAULT_CLASS_CONF_THRESHOLD,
    "animal": DEFAULT_CLASS_CONF_THRESHOLD,
    "vertical_obstacle": DEFAULT_CLASS_CONF_THRESHOLD,
    "temporary_obstacle": DEFAULT_CLASS_CONF_THRESHOLD,
    "bench": DEFAULT_CLASS_CONF_THRESHOLD,
    "traffic_light": DEFAULT_CLASS_CONF_THRESHOLD,
    "traffic_sign": DEFAULT_CLASS_CONF_THRESHOLD,
    "bus_taxi_stop": DEFAULT_CLASS_CONF_THRESHOLD,
    "sidewalk": DEFAULT_CLASS_CONF_THRESHOLD,
    "braille_guide_blocks": DEFAULT_CLASS_CONF_THRESHOLD,
    "roadway": DEFAULT_CLASS_CONF_THRESHOLD,
    "alley": DEFAULT_CLASS_CONF_THRESHOLD,
    "crosswalk": DEFAULT_CLASS_CONF_THRESHOLD,
    "bike_lane": DEFAULT_CLASS_CONF_THRESHOLD,
    "caution_zone": DEFAULT_CLASS_CONF_THRESHOLD,
}
MAX_DETECTIONS = 80
MAX_PREVIEW_BOXES = 20


# 알림은 같은 객체/같은 종류가 너무 자주 반복되지 않도록 cooldown을 둡니다.
ALERT_COOLDOWN_SEC = 4.0
ALERTS_PER_INFERENCE = 1


# 보행 경로 ROI입니다. 좌표는 512x288 프레임 기준 정규화 x/y 값입니다.
# 아래쪽은 가까운 보행 공간, 위쪽은 먼 전방 공간이라 사다리꼴로 좁아집니다.
PATH_ROI_POLYGON = [
    (0.44, 0.88),
    (0.56, 0.88),
    (0.64, 0.98),
    (0.36, 0.98),
]

# 차량 ROI입니다. 차량은 옆에서 들어오는 경우가 많아 보행 경로 ROI보다 넓게 봅니다.
VEHICLE_ROI_POLYGON = [
    (0.41, 0.70),
    (0.59, 0.70),
    (0.70, 0.98),
    (0.30, 0.98),
]


# 사람/장애물/차량 안전 판단 기준입니다.
FORWARD_ZONE_X1 = 0.35
FORWARD_ZONE_X2 = 0.65

STATIC_NEAR_AREA_RATIO = 0.0015
STATIC_CONFIRM_FRAMES = 3
ROI_BOTTOM_OVERLAP_RATIO = 0.10
STATIC_VEHICLE_SUPPRESS_OVERLAP_RATIO = 0.35

# 차량 이동 위험은 두 가지로 봅니다.
# 1. 정면 접근: ROI 안에서 bbox 면적이 빠르게 커지고 하단이 아래로 내려오는 경우
# 2. 측면 진입: bbox 중심이 화면 중앙 쪽으로 움직이며 충분히 가까운 경우
MOVING_NEAR_AREA_RATIO = 0.05
MOVING_APPROACH_SCALE = 1.18
MOVING_FAST_AREA_GROWTH_PER_SEC = 0.45
MOVING_APPROACH_WINDOW_SEC = 1.5
MOVING_TRACK_MAX_GAP_SEC = 1.2
MOVING_MIN_OBSERVATIONS = 2
MOVING_MATCH_DISTANCE = 0.20
MOVING_FRONTAL_MIN_BOTTOM_SHIFT = 0.02
MOVING_LATERAL_MIN_AREA_RATIO = 0.025
MOVING_LATERAL_MIN_BOTTOM_Y = 0.74
MOVING_LATERAL_MIN_X_SHIFT = 0.04

PERSON_NEAR_AREA_RATIO = 0.16
PERSON_NEAR_BOTTOM_Y = 0.82
PERSON_NEAR_WIDTH_RATIO = 0.18


# 지면 mask 기반 보행 경로 판단 기준입니다.
# PATH_ROI_POLYGON 안에서 면적이 가장 넓은 지면 라벨을 현재 경로 상태로 봅니다.
SURFACE_CONFIRM_FRAMES = 4
SURFACE_MIN_DOMINANT_RATIO = 0.08
SURFACE_BRAILLE_PRIORITY_RATIO = 0.03
