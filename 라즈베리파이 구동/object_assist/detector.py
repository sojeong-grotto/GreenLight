from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from config import (
    CLASS_CONF_THRESHOLDS,
    LETTERBOX_COLOR,
    DETECTION_CONF_THRESHOLD,
    MAX_DETECTIONS,
    MODEL_INPUT_HEIGHT,
    MODEL_INPUT_WIDTH,
    OBJECT_LABEL_PATH,
    OBJECT_MODEL_PATH,
    TFLITE_BACKEND,
    TFLITE_DISABLE_XNNPACK,
    TFLITE_NUM_THREADS,
    USE_MOCK_DETECTOR_IF_MODEL_MISSING,
)


# detector.py는 객체 TFLite YOLO 모델을 실행하고 bbox Detection 목록을 만듭니다.
# 현재 런타임은 마스크를 쓰지 않으므로 bbox 후처리만 남겨 추론 뒤 연산량을 줄입니다.


@dataclass
class Detection:
    class_id: int
    label: str
    score: float
    box: tuple[float, float, float, float] | None = None  # 카메라 프레임 기준 정규화 좌표 x1,y1,x2,y2
    mask: np.ndarray | None = None  # 카메라 프레임 크기 HxW, 값은 0/1


class MockDetector:
    # 모델 파일이 아직 없을 때도 카메라, preview, TTS 흐름을 확인하기 위한 빈 detector입니다.
    def __init__(self, labels):
        self.labels = labels

    def predict(self, frame_bgr):
        return []


class TFLiteYoloSegDetector:
    def __init__(
        self,
        model_path=OBJECT_MODEL_PATH,
        label_path=OBJECT_LABEL_PATH,
        input_size=(MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT),
        enable_masks=False,
    ):
        self.model_path = Path(model_path)
        self.labels = load_labels(label_path)
        self.input_w, self.input_h = input_size
        self.enable_masks = enable_masks

        if not self.model_path.exists():
            if USE_MOCK_DETECTOR_IF_MODEL_MISSING:
                self.mock = MockDetector(self.labels)
                self.interpreter = None
                print("모델 파일이 없어 mock detector로 실행합니다:", self.model_path)
                return
            raise FileNotFoundError(f"TFLite 모델이 없습니다: {self.model_path}")

        self.mock = None
        self.interpreter, self.backend_name = create_allocated_interpreter(str(self.model_path))
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_w, self.input_h = infer_input_size(self.input_details[0], fallback=(self.input_w, self.input_h))

        print("TFLite backend:", self.backend_name)
        print("TFLite input:", self.input_details[0]["shape"], self.input_details[0]["dtype"], self.input_details[0].get("quantization"))
        print("TFLite outputs:", [(d["shape"].tolist(), str(d["dtype"]), d.get("quantization")) for d in self.output_details])

    def predict(self, frame_bgr):
        # 한 프레임을 모델 입력으로 변환하고, TFLite 추론 후 Detection 리스트로 반환합니다.
        # 좌표는 detector 내부에서 다시 원본 카메라 프레임 기준 0~1 정규화 좌표로 복원됩니다.
        if self.mock is not None:
            return self.mock.predict(frame_bgr)

        input_tensor, preprocess_meta = preprocess(frame_bgr, self.input_w, self.input_h, self.input_details[0])
        self.interpreter.set_tensor(self.input_details[0]["index"], input_tensor)
        self.interpreter.invoke()
        outputs = [dequantize_output(self.interpreter.get_tensor(o["index"]), o) for o in self.output_details]

        detections = parse_ultralytics_outputs(
            outputs,
            self.labels,
            preprocess_meta,
            self.enable_masks,
        )
        return detections


def create_allocated_interpreter(model_path):
    # 라즈베리파이에서는 TFLite delegate/XNNPACK 조합에 따라 모델이 열리지 않는 경우가 있습니다.
    # 여러 interpreter 설정을 순서대로 시도해 가장 먼저 성공하는 backend를 사용합니다.
    errors = []
    for name, factory in interpreter_factories(model_path):
        try:
            interpreter = factory()
            interpreter.allocate_tensors()
            return interpreter, name
        except Exception as exc:
            errors.append((name, repr(exc)))

    detail = "\n".join(f"- {name}: {err}" for name, err in errors)
    raise RuntimeError(
        "TFLite 모델 초기화에 모두 실패했습니다.\n"
        "INT8 YOLO-seg 모델의 hybrid quantization/TRANSPOSE_CONV가 현재 런타임과 맞지 않을 수 있습니다.\n"
        "아래 실패 내역을 확인하세요.\n"
        f"{detail}"
    )


def interpreter_factories(model_path):
    factories = []
    preferred = TFLITE_BACKEND.strip().lower()

    try:
        from ai_edge_litert.interpreter import Interpreter

        def make_litert(name):
            def factory():
                kwargs = interpreter_kwargs(model_path)
                return Interpreter(**kwargs)

            factories.append((name, factory))

        # LiteRT 2.x exposes the legacy Interpreter API, but its OpResolverType
        # enum is not accepted by every build. Keep LiteRT as the first backend
        # and let the older runtimes handle resolver variants below.
        make_litert("ai_edge_litert:default")
    except Exception:
        pass

    try:
        from tflite_runtime.interpreter import Interpreter, OpResolverType

        def make_tflite_runtime(name, resolver=None):
            def factory():
                kwargs = interpreter_kwargs(model_path)
                if resolver is not None:
                    kwargs["experimental_op_resolver_type"] = resolver
                return Interpreter(**kwargs)

            factories.append((name, factory))

        without_delegates = getattr(OpResolverType, "BUILTIN_WITHOUT_DEFAULT_DELEGATES", None)
        builtin_ref = getattr(OpResolverType, "BUILTIN_REF", None)
        builtin = getattr(OpResolverType, "BUILTIN", None)

        if TFLITE_DISABLE_XNNPACK and without_delegates is not None:
            make_tflite_runtime("tflite_runtime:no_default_delegates", without_delegates)
        if builtin_ref is not None:
            make_tflite_runtime("tflite_runtime:builtin_ref", builtin_ref)
        if builtin is not None:
            make_tflite_runtime("tflite_runtime:builtin", builtin)
        make_tflite_runtime("tflite_runtime:default", None)
    except Exception:
        pass

    try:
        import tensorflow as tf

        def make_tensorflow(name, resolver=None):
            def factory():
                kwargs = interpreter_kwargs(model_path)
                if resolver is not None:
                    kwargs["experimental_op_resolver_type"] = resolver
                return tf.lite.Interpreter(**kwargs)

            factories.append((name, factory))

        resolver_type = getattr(tf.lite.experimental, "OpResolverType", None)
        if resolver_type is not None:
            without_delegates = getattr(resolver_type, "BUILTIN_WITHOUT_DEFAULT_DELEGATES", None)
            builtin_ref = getattr(resolver_type, "BUILTIN_REF", None)
            builtin = getattr(resolver_type, "BUILTIN", None)
            if TFLITE_DISABLE_XNNPACK and without_delegates is not None:
                make_tensorflow("tensorflow:no_default_delegates", without_delegates)
            if builtin_ref is not None:
                make_tensorflow("tensorflow:builtin_ref", builtin_ref)
            if builtin is not None:
                make_tensorflow("tensorflow:builtin", builtin)
        make_tensorflow("tensorflow:default", None)
    except Exception:
        pass

    if not factories:
        raise ImportError("ai_edge_litert, tflite_runtime, tensorflow 중 설치된 TFLite interpreter가 없습니다.")
    if preferred != "auto":
        selected = [(name, factory) for name, factory in factories if name == preferred or name.startswith(preferred + ":")]
        if selected:
            return selected
        print(f"requested TFLite backend not found, using auto: {TFLITE_BACKEND}")
    return factories


def interpreter_kwargs(model_path):
    kwargs = {"model_path": model_path}
    if TFLITE_NUM_THREADS > 0:
        kwargs["num_threads"] = TFLITE_NUM_THREADS
    return kwargs


def load_labels(label_path):
    # 라벨 파일은 class id 순서와 정확히 일치해야 합니다. 주석이나 빈 줄은 넣지 않는 것이 안전합니다.
    path = Path(label_path)
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def infer_input_size(input_detail, fallback):
    # 대부분의 Ultralytics TFLite는 NHWC [1, height, width, 3]입니다.
    # 혹시 다른 축 순서로 export된 모델이 들어와도 가능한 범위에서 입력 크기를 추정합니다.
    shape = np.asarray(input_detail["shape"]).astype(int).tolist()
    if len(shape) == 4:
        # Ultralytics TFLite exports are normally NHWC: [1, height, width, 3].
        if shape[-1] in {1, 3}:
            return int(shape[2]), int(shape[1])
        # Rare NCHW fallback.
        if shape[1] in {1, 3}:
            return int(shape[3]), int(shape[2])
    return fallback


def preprocess(frame_bgr, width, height, input_detail):
    # Camera frames stay 16:9 for display and safety logic. The model receives a
    # letterboxed 512x512 tensor, matching the training/export pipeline.
    img, meta = letterbox(frame_bgr, width, height)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    dtype = input_detail["dtype"]

    if dtype == np.float32:
        tensor = img.astype(np.float32) / 255.0
    elif dtype in {np.uint8, np.int8}:
        tensor = quantize_input(img, input_detail, dtype)
    else:
        tensor = img.astype(dtype)

    return np.expand_dims(tensor, axis=0), meta


def quantize_input(img_uint8, input_detail, dtype):
    scale, zero_point = input_detail.get("quantization", (0.0, 0))
    if not scale:
        return img_uint8.astype(dtype)

    img_float = img_uint8.astype(np.float32) / 255.0
    q = np.round(img_float / float(scale) + int(zero_point))
    if dtype == np.int8:
        q = np.clip(q, -128, 127)
    else:
        q = np.clip(q, 0, 255)
    return q.astype(dtype)


def dequantize_output(output, output_detail):
    # INT8/UINT8 출력은 scale/zero_point를 이용해 float 값으로 되돌린 뒤 후처리합니다.
    scale, zero_point = output_detail.get("quantization", (0.0, 0))
    if not scale:
        return output
    return (output.astype(np.float32) - int(zero_point)) * float(scale)


def letterbox(frame_bgr, width, height):
    # 학습과 동일하게 비율을 유지한 채 512x512 캔버스에 넣습니다.
    # 반환되는 meta는 모델 좌표를 원본 프레임 좌표로 되돌릴 때 사용합니다.
    src_h, src_w = frame_bgr.shape[:2]
    scale = min(width / src_w, height / src_h)
    resized_w = int(round(src_w * scale))
    resized_h = int(round(src_h * scale))
    resized = cv2.resize(frame_bgr, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((height, width, 3), LETTERBOX_COLOR, dtype=np.uint8)
    pad_x = (width - resized_w) // 2
    pad_y = (height - resized_h) // 2
    canvas[pad_y : pad_y + resized_h, pad_x : pad_x + resized_w] = resized
    meta = {
        "input_size": (width, height),
        "source_size": (src_w, src_h),
        "scale": scale,
        "pad": (pad_x, pad_y),
        "resized_size": (resized_w, resized_h),
    }
    return canvas, meta


def input_box_to_frame_box(box, meta):
    # letterbox padding을 제거하고, 모델 입력 좌표를 카메라 프레임 좌표로 복원합니다.
    x1, y1, x2, y2 = box
    input_w, input_h = meta["input_size"]
    src_w, src_h = meta["source_size"]
    pad_x, pad_y = meta["pad"]
    scale = meta["scale"]

    px1 = x1 * input_w
    py1 = y1 * input_h
    px2 = x2 * input_w
    py2 = y2 * input_h

    fx1 = (px1 - pad_x) / scale / src_w
    fy1 = (py1 - pad_y) / scale / src_h
    fx2 = (px2 - pad_x) / scale / src_w
    fy2 = (py2 - pad_y) / scale / src_h
    return clamp_box((fx1, fy1, fx2, fy2))


def input_mask_to_frame_mask(mask, meta):
    input_w, input_h = meta["input_size"]
    src_w, src_h = meta["source_size"]
    pad_x, pad_y = meta["pad"]
    resized_w, resized_h = meta["resized_size"]

    if mask.shape[:2] != (input_h, input_w):
        mask = cv2.resize(mask.astype(np.uint8), (input_w, input_h), interpolation=cv2.INTER_NEAREST)
    unpadded = mask[pad_y : pad_y + resized_h, pad_x : pad_x + resized_w]
    return cv2.resize(unpadded.astype(np.uint8), (src_w, src_h), interpolation=cv2.INTER_NEAREST)


def clamp_box(box):
    x1, y1, x2, y2 = box
    return (
        max(0.0, min(1.0, x1)),
        max(0.0, min(1.0, y1)),
        max(0.0, min(1.0, x2)),
        max(0.0, min(1.0, y2)),
    )


def confidence_threshold_for_label(label):
    return CLASS_CONF_THRESHOLDS.get(label, DETECTION_CONF_THRESHOLD)


def keep_detection(label, score):
    return float(score) >= confidence_threshold_for_label(label)


def parse_ultralytics_outputs(outputs, labels, preprocess_meta, enable_masks=False):
    # 현재 변환 노트북은 NMS 포함 TFLite 출력을 만들도록 설정되어 있습니다.
    # 이 파서는 [x1, y1, x2, y2, score, class_id, ...] 형태의 첫 6개 값만 사용합니다.
    detection_output = find_detection_output(outputs)
    if detection_output is None:
        shapes = [tuple(o.shape) for o in outputs]
        raise NotImplementedError(
            "지원하지 않는 TFLite YOLO 출력 형태입니다. "
            f"모델 출력 shapes={shapes}, preprocess={preprocess_meta}"
        )

    mask_proto = find_mask_proto_output(outputs)
    detections = []
    rows = detection_output[0] if detection_output.ndim == 3 else detection_output
    for row in rows:
        parsed = parse_nms_detection_row(row, labels, preprocess_meta, mask_proto, enable_masks)
        if parsed is None:
            continue
        detections.append(parsed)
        if len(detections) >= MAX_DETECTIONS:
            break
    detections.sort(key=lambda det: det.score, reverse=True)
    return detections


def find_detection_output(outputs):
    for output in outputs:
        arr = np.asarray(output)
        if arr.ndim == 3 and arr.shape[0] == 1 and arr.shape[-1] >= 6:
            return arr
        if arr.ndim == 2 and arr.shape[-1] >= 6:
            return arr
    return None


def find_mask_proto_output(outputs):
    for output in outputs:
        arr = np.asarray(output)
        if arr.ndim == 4 and arr.shape[0] == 1 and arr.shape[-1] > 1:
            return arr[0]
    return None


def parse_nms_detection_row(row, labels, preprocess_meta, mask_proto=None, enable_masks=False):
    x1, y1, x2, y2 = [float(v) for v in row[:4]]
    score = float(row[4])
    class_id = int(round(float(row[5])))
    if score <= 0 or class_id < 0 or class_id >= len(labels):
        return None

    label = labels[class_id]
    if not keep_detection(label, score):
        return None

    box = normalize_box((x1, y1, x2, y2), preprocess_meta)
    if box is None:
        return None

    mask = None
    if enable_masks and mask_proto is not None and row.shape[0] > 6:
        mask = build_detection_mask(mask_proto, row[6:], (x1, y1, x2, y2), preprocess_meta)

    return Detection(class_id=class_id, label=label, score=score, box=box, mask=mask)


def normalize_box(box, preprocess_meta):
    x1, y1, x2, y2 = box
    if max(abs(x1), abs(y1), abs(x2), abs(y2)) > 2.0:
        input_w, input_h = preprocess_meta["input_size"]
        box = (x1 / input_w, y1 / input_h, x2 / input_w, y2 / input_h)

    x1, y1, x2, y2 = input_box_to_frame_box(box, preprocess_meta)
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def build_detection_mask(mask_proto, coeffs, input_box, preprocess_meta):
    if coeffs.shape[0] != mask_proto.shape[-1]:
        return None

    proto_h, proto_w, proto_c = mask_proto.shape
    logits = np.tensordot(mask_proto.reshape(-1, proto_c), coeffs.astype(np.float32), axes=(1, 0))
    mask = sigmoid(logits).reshape(proto_h, proto_w)

    x1, y1, x2, y2 = normalize_input_box(input_box, preprocess_meta["input_size"])
    px1 = int(np.floor(x1 * proto_w))
    py1 = int(np.floor(y1 * proto_h))
    px2 = int(np.ceil(x2 * proto_w))
    py2 = int(np.ceil(y2 * proto_h))
    px1 = max(0, min(proto_w - 1, px1))
    py1 = max(0, min(proto_h - 1, py1))
    px2 = max(px1 + 1, min(proto_w, px2))
    py2 = max(py1 + 1, min(proto_h, py2))

    cropped = np.zeros_like(mask, dtype=np.uint8)
    cropped[py1:py2, px1:px2] = (mask[py1:py2, px1:px2] >= 0.5).astype(np.uint8)
    return input_mask_to_frame_mask(cropped, preprocess_meta)


def normalize_input_box(box, input_size):
    x1, y1, x2, y2 = box
    if max(abs(x1), abs(y1), abs(x2), abs(y2)) > 2.0:
        input_w, input_h = input_size
        return x1 / input_w, y1 / input_h, x2 / input_w, y2 / input_h
    return x1, y1, x2, y2


def sigmoid(x):
    x = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x))
