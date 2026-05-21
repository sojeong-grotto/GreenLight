import argparse
from collections import deque
from pathlib import Path
import threading
import time

import cv2
import numpy as np

from camera import Camera, crop_to_aspect_and_resize
from config import (
    ALERT_COOLDOWN_SEC,
    ALERTS_PER_INFERENCE,
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    MAX_PREVIEW_BOXES,
    OBJECT_INFERENCE_FPS,
    OBJECT_LABEL_PATH,
    OBJECT_MODEL_PATH,
    PREVIEW_FPS,
    SHOW_PREVIEW,
    SURFACE_INFERENCE_FPS,
    SURFACE_LABEL_PATH,
    SURFACE_MODEL_PATH,
)
from detector import TFLiteYoloSegDetector
from safety import (
    MOVING_RISK_LABELS,
    PERSON_LABELS,
    STATIC_OBSTACLE_LABELS,
    ObjectSafetyAnalyzer,
    SurfaceSafetyAnalyzer,
    alert_roi_polygon,
    vehicle_roi_polygon,
)
from tts import AlertSpeaker


WINDOW_NAME = "ObjectAssist"
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
POLYGON_POINT_CACHE = {}


def run(args):
    """프로그램 진입점입니다.

    object_assist는 두 가지 실행만 남깁니다.
    1. 카메라 실행: 실제 보행 보조 흐름입니다.
    2. 동영상 실행: 같은 전처리/추론/알림 로직을 영상으로 확인하는 흐름입니다.

    벤치마크, 모델 프로브, 스무딩 실험 코드는 제거하고 런타임에 필요한 경로만
    유지해 추론 루프에서 불필요한 분기와 후처리를 줄였습니다.
    """
    if args.list_videos:
        print_available_videos()
        return

    detector = TFLiteYoloSegDetector(model_path=OBJECT_MODEL_PATH, label_path=OBJECT_LABEL_PATH)
    surface_detector = None
    if args.surface_mask:
        surface_detector = TFLiteYoloSegDetector(
            model_path=SURFACE_MODEL_PATH,
            label_path=SURFACE_LABEL_PATH,
            enable_masks=True,
        )
    analyzer = ObjectSafetyAnalyzer()
    surface_analyzer = SurfaceSafetyAnalyzer() if surface_detector is not None else None
    alert_gate = AlertGate()
    alert_log = AlertOverlayLog(args.alert_log_size, args.alert_log_sec)
    speaker = None if args.no_speak else AlertSpeaker()
    source = open_source(args)
    source_fps = get_source_fps(source, args.preview_fps)

    print_runtime_info(args, source_fps)
    speak_startup(speaker)

    try:
        if args.video and args.preview:
            run_video_preview(args, source, source_fps, detector, surface_detector, analyzer, surface_analyzer, alert_gate, alert_log, speaker)
        else:
            run_camera_loop(args, source, source_fps, detector, surface_detector, analyzer, surface_analyzer, alert_gate, alert_log, speaker)
    except KeyboardInterrupt:
        pass
    finally:
        close_source(source)
        cv2.destroyAllWindows()


def run_camera_loop(args, source, source_fps, detector, surface_detector, analyzer, surface_analyzer, alert_gate, alert_log, speaker):
    """카메라 또는 preview 없는 동영상용 단순 추론 루프입니다.

    카메라에서는 프레임을 읽고, 설정한 추론 FPS 간격에 맞을 때만 모델을 실행합니다.
    preview가 켜져 있으면 추론이 끝난 프레임만 그립니다. 이렇게 하면 박스가
    다른 프레임 위에 밀려 보이지 않고, 오버레이 연산도 추론 프레임에만 붙습니다.
    """
    metrics = empty_metrics(args.fps)
    last_infer_start = 0.0
    last_surface_infer_start = 0.0
    surface_done_times = deque()
    surface_detections = []
    surface_events = []
    last_status_time = time.monotonic()
    status_frames = 0
    status_inferences = 0

    while True:
        frame = read_frame(source)
        if frame is None:
            break

        status_frames += 1
        video_time_sec = get_source_time_sec(source, source_fps)
        now = time.monotonic()
        if not should_run_inference(now, last_infer_start, args.fps, args.fast):
            sleep_for_camera_loop(args, last_infer_start)
            continue

        infer_start = time.monotonic()
        last_infer_start = infer_start
        detections = detector.predict(frame)
        events = analyzer.analyze(detections)
        infer_done = time.monotonic()
        status_inferences += 1
        metrics = metrics_from_times(infer_start, infer_done, args.fps)

        emit_alerts(events, speaker, alert_gate, video_time_sec, alert_log)

        if surface_detector is not None and should_run_inference(infer_start, last_surface_infer_start, args.surface_fps, args.fast):
            last_surface_infer_start = time.monotonic()
            surface_detections = surface_detector.predict(frame)
            update_recent_times(surface_done_times, time.monotonic())
            metrics["surface_fps"] = recent_fps_from_times(surface_done_times)
            metrics["surface_target_fps"] = args.surface_fps
            surface_events = surface_analyzer.analyze(surface_detections) if surface_analyzer is not None else []
            metrics["surface_status"] = format_surface_status(surface_analyzer)
            emit_alerts(surface_events, speaker, alert_gate, video_time_sec, alert_log)

        if args.preview:
            preview = frame.copy()
            draw_preview(preview, detections, surface_detections, combined_events(events, surface_events), metrics, args, alert_log)
            show_preview(preview, args.window_scale)
            key = cv2.waitKey(max(1, args.delay_ms)) & 0xFF
            if key == ord("q"):
                break

        last_status_time, status_frames, status_inferences = maybe_print_status(
            last_status_time,
            status_frames,
            status_inferences,
            args,
            metrics,
            surface_analyzer,
            combined_events(events, surface_events),
            video_time_sec,
        )


def run_video_preview(args, source, source_fps, detector, surface_detector, analyzer, surface_analyzer, alert_gate, alert_log, speaker):
    """동영상 확인용 preview 루프입니다.

    동영상 preview에서는 화면 출력과 추론을 분리합니다. 메인 루프는 영상을
    preview FPS에 맞춰 읽고, 별도 worker는 항상 가장 최신 프레임만 추론합니다.
    오래된 프레임을 큐에 쌓지 않기 때문에 느린 추론 환경에서도 지연이 계속
    누적되지 않습니다.
    """
    worker = AsyncInferenceWorker(detector, analyzer, args.fps)
    worker.start()
    surface_worker = None
    if surface_detector is not None:
        surface_worker = AsyncInferenceWorker(surface_detector, surface_analyzer, args.surface_fps)
        surface_worker.start()

    display_frame = None
    detections = []
    surface_detections = []
    surface_events = []
    events = []
    metrics = empty_metrics(args.fps)
    last_result_id = -1
    last_surface_result_id = -1
    last_status_time = time.monotonic()
    status_frames = 0
    status_inferences = 0

    try:
        while True:
            loop_start = time.monotonic()
            frame = read_frame(source)
            if frame is None:
                break

            video_time_sec = get_source_time_sec(source, source_fps)
            status_frames += 1
            worker.submit(frame, video_time_sec)
            if surface_worker is not None:
                surface_worker.submit(frame, video_time_sec)

            result = worker.latest_result()
            if result and result["id"] != last_result_id:
                detections = result["detections"]
                events = result["events"]
                metrics = preserve_surface_metrics(metrics_from_result(result, worker.recent_fps(), args.fps), metrics)
                display_frame = result["frame"]
                last_result_id = result["id"]
                status_inferences += 1
                emit_alerts(events, speaker, alert_gate, result.get("video_time_sec"), alert_log)

            surface_result = surface_worker.latest_result() if surface_worker is not None else None
            if surface_result and surface_result["id"] != last_surface_result_id:
                surface_detections = surface_result["detections"]
                surface_events = surface_result["events"]
                last_surface_result_id = surface_result["id"]
                metrics["surface_fps"] = surface_worker.recent_fps()
                metrics["surface_target_fps"] = args.surface_fps
                metrics["surface_status"] = format_surface_status(surface_analyzer)
                emit_alerts(surface_events, speaker, alert_gate, surface_result.get("video_time_sec"), alert_log)

            preview = (display_frame if display_frame is not None else frame).copy()
            draw_preview(preview, detections, surface_detections, combined_events(events, surface_events), metrics, args, alert_log)
            show_preview(preview, args.window_scale)

            key = cv2.waitKey(max(1, args.delay_ms)) & 0xFF
            if key == ord("q"):
                break
            seek_delta = seek_delta_from_key(key, args)
            if seek_delta is not None:
                seek_video(source, seek_delta, source_fps)
                worker.reset()
                analyzer.reset()
                alert_gate.reset()
                alert_log.reset()
                display_frame = None
                detections = []
                surface_detections = []
                surface_events = []
                events = []
                last_result_id = -1
                last_surface_result_id = -1
                if surface_worker is not None:
                    surface_worker.reset()
                if surface_analyzer is not None:
                    surface_analyzer.reset()
            elif not args.fast:
                skip_video_frames_for_preview(source, source_fps, args.preview_fps)
                sleep_to_preview_fps(loop_start, args.preview_fps)

            last_status_time, status_frames, status_inferences = maybe_print_status(
                last_status_time,
                status_frames,
                status_inferences,
                args,
                metrics,
                surface_analyzer,
                combined_events(events, surface_events),
                video_time_sec,
            )
    finally:
        worker.stop()
        if surface_worker is not None:
            surface_worker.stop()


class AsyncInferenceWorker:
    """동영상 preview에서 최신 프레임만 추론하는 worker입니다.

    입력 큐를 길게 두면 추론이 느릴 때 과거 프레임을 계속 처리하게 됩니다.
    여기서는 pending 슬롯 하나만 두고 새 프레임이 오면 이전 pending을 덮어씁니다.
    그래서 추론 FPS가 낮아도 결과는 가능한 최신 화면에 가깝게 유지됩니다.
    """

    def __init__(self, detector, analyzer, fps):
        self.detector = detector
        self.analyzer = analyzer
        self.fps = fps
        self.condition = threading.Condition()
        self.pending = None
        self.latest = None
        self.result_id = 0
        self.done_times = deque()
        self.last_infer_start = 0.0
        self.running = False
        self.thread = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        with self.condition:
            self.running = False
            self.condition.notify_all()
        if self.thread is not None:
            self.thread.join(timeout=1.0)

    def submit(self, frame, video_time_sec):
        with self.condition:
            self.pending = {
                "frame": frame,
                "submitted_at": time.monotonic(),
                "video_time_sec": video_time_sec,
            }
            self.condition.notify()

    def latest_result(self):
        with self.condition:
            return self.latest

    def recent_fps(self):
        with self.condition:
            now = time.monotonic()
            while self.done_times and now - self.done_times[0] > 2.0:
                self.done_times.popleft()
            if len(self.done_times) < 2:
                return float(len(self.done_times))
            elapsed = max(self.done_times[-1] - self.done_times[0], 1e-6)
            return (len(self.done_times) - 1) / elapsed

    def reset(self):
        with self.condition:
            self.pending = None
            self.latest = None
            self.result_id += 1
            self.done_times.clear()
            self.last_infer_start = 0.0

    def _loop(self):
        while True:
            with self.condition:
                while self.running and self.pending is None:
                    self.condition.wait(timeout=0.1)
                if not self.running:
                    return
                pending = self.pending
                self.pending = None

            wait_for_inference_slot(self.last_infer_start, self.fps)
            pending = self._take_newer_pending(pending)

            frame = pending["frame"]
            infer_start = time.monotonic()
            self.last_infer_start = infer_start
            detections = self.detector.predict(frame)
            events = self.analyzer.analyze(detections) if self.analyzer is not None else []
            infer_done = time.monotonic()

            with self.condition:
                self.result_id += 1
                self.latest = {
                    "id": self.result_id,
                    "frame": frame,
                    "detections": detections,
                    "events": events,
                    "submitted_at": pending["submitted_at"],
                    "started_at": infer_start,
                    "done_at": infer_done,
                    "video_time_sec": pending.get("video_time_sec"),
                    "latency_ms": (infer_done - pending["submitted_at"]) * 1000.0,
                    "infer_ms": (infer_done - infer_start) * 1000.0,
                }
                self.done_times.append(infer_done)
                while self.done_times and infer_done - self.done_times[0] > 2.0:
                    self.done_times.popleft()

    def _take_newer_pending(self, pending):
        with self.condition:
            if self.pending is not None:
                pending = self.pending
                self.pending = None
        return pending


class AlertGate:
    """같은 객체에 대한 같은 알림이 너무 짧게 반복되지 않도록 막습니다."""

    def __init__(self, cooldown_sec=ALERT_COOLDOWN_SEC):
        self.cooldown_sec = cooldown_sec
        self.last_emitted = {}

    def allow(self, event):
        now = time.monotonic()
        key = event_cooldown_key(event)
        last = self.last_emitted.get(key, 0.0)
        if now - last < self.cooldown_sec:
            return False
        self.last_emitted[key] = now
        return True

    def reset(self):
        self.last_emitted.clear()


class AlertOverlayLog:
    """화면 왼쪽 위에 최근 알림을 로그처럼 남기기 위한 작은 저장소입니다."""

    def __init__(self, max_items=5, ttl_sec=12.0):
        self.max_items = max(1, max_items)
        self.ttl_sec = max(1.0, ttl_sec)
        self.items = deque()

    def add(self, event, video_time_sec=None):
        key = f"gnd:{event.key}" if is_surface_alert(event) else event.key
        self.items.appendleft(
            {
                "key": key,
                "time_text": format_video_time(video_time_sec) if video_time_sec is not None else "--:--.-",
                "created_at": time.monotonic(),
            }
        )
        while len(self.items) > self.max_items:
            self.items.pop()

    def visible_items(self):
        now = time.monotonic()
        self.items = deque(item for item in self.items if now - item["created_at"] <= self.ttl_sec)
        return list(self.items)

    def reset(self):
        self.items.clear()


def should_run_inference(now, last_infer_time, fps, fast):
    if fast or last_infer_time == 0.0:
        return True
    return now - last_infer_time >= 1.0 / max(fps, 0.1)


def wait_for_inference_slot(last_infer_time, fps):
    if last_infer_time <= 0.0:
        return
    wait = 1.0 / max(fps, 0.1) - (time.monotonic() - last_infer_time)
    if wait > 0:
        time.sleep(wait)


def sleep_for_camera_loop(args, last_infer_time):
    if args.fast or last_infer_time <= 0.0:
        return
    # 너무 촘촘하게 busy-loop를 돌지 않도록 아주 짧게 쉽니다.
    time.sleep(min(0.01, 1.0 / max(args.fps, 1.0)))


def emit_alerts(events, speaker, alert_gate, video_time_sec=None, alert_log=None):
    emitted = 0
    for event in events:
        if is_suppressed_alert(event):
            continue
        if emitted >= ALERTS_PER_INFERENCE:
            break
        if not alert_gate.allow(event):
            continue
        category = "SURFACE" if is_surface_alert(event) else "ALERT"
        print(f"{format_alert_time(video_time_sec)}[{category}] {event.key}: {event.message}")
        if alert_log is not None:
            alert_log.add(event, video_time_sec)
        if speaker is not None:
            speaker.speak(event.message, key=event.key, force=True)
        emitted += 1


def combined_events(*event_groups):
    events = []
    for group in event_groups:
        events.extend(group or [])
    events.sort(key=lambda event: event.priority, reverse=True)
    return events


def is_surface_alert(event):
    return event.object_id is not None and event.object_id.startswith("surface:")


def is_suppressed_alert(event):
    # 차량을 고정 장애물로 알리는 경로는 안정성이 낮아서 사용하지 않습니다.
    # 혹시 이전 로직이나 외부 경로에서 같은 메시지가 들어와도 출력하지 않습니다.
    return event.key == "static_obstacle" and "차량 장애물" in event.message


def maybe_print_status(last_time, frames, inferences, args, metrics, surface_analyzer, events, video_time_sec=None):
    if not args.terminal_status:
        return last_time, frames, inferences

    now = time.monotonic()
    if now - last_time < args.status_interval_sec:
        return last_time, frames, inferences

    elapsed = max(now - last_time, 1e-6)
    preview_fps = frames / elapsed
    infer_fps = inferences / elapsed
    print(
        f"{format_alert_time(video_time_sec)}preview={preview_fps:.1f}fps "
        f"infer={infer_fps:.1f}/{args.fps:.1f}fps "
        f"latency={format_ms(metrics.get('latency_ms'))} "
        f"ground={format_surface_status(surface_analyzer)}"
    )
    return now, 0, 0


def format_surface_status(surface_analyzer):
    if surface_analyzer is None or surface_analyzer.last_dominant is None:
        return "none"
    label, ratio = surface_analyzer.last_dominant
    return f"{label}:{ratio:.2f}"


def summarize_labels(detections, limit=6):
    counts = {}
    for det in detections:
        counts[det.label] = counts.get(det.label, 0) + 1
    if not counts:
        return "none"
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    parts = [f"{label}:{count}" for label, count in ordered[:limit]]
    if len(ordered) > limit:
        parts.append(f"+{len(ordered) - limit}")
    return ",".join(parts)


def open_source(args):
    if args.video:
        path = resolve_video_path(args.video)
        if not path.exists():
            raise FileNotFoundError(f"동영상 파일이 없습니다: {path}")
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"동영상을 열 수 없습니다: {path}")
        if args.start_sec > 0:
            cap.set(cv2.CAP_PROP_POS_MSEC, args.start_sec * 1000.0)
        print("video:", path)
        return cap
    return Camera(camera_index=args.camera).start()


def read_frame(source):
    if isinstance(source, Camera):
        return source.read()
    ok, frame = source.read()
    if not ok:
        return None
    return crop_to_aspect_and_resize(frame)


def close_source(source):
    if isinstance(source, Camera):
        source.stop()
    else:
        source.release()


def is_video_source(source):
    return not isinstance(source, Camera)


def get_source_fps(source, fallback):
    if not is_video_source(source):
        return fallback
    fps = source.get(cv2.CAP_PROP_FPS)
    return fps if fps and fps > 0 else fallback


def get_source_time_sec(source, source_fps):
    if not is_video_source(source):
        return None
    msec = source.get(cv2.CAP_PROP_POS_MSEC)
    if msec and msec > 0:
        return msec / 1000.0
    frame_index = max(0, int(source.get(cv2.CAP_PROP_POS_FRAMES) or 0) - 1)
    return frame_index / max(source_fps, 0.1)


def seek_video(source, delta_sec, source_fps):
    if not is_video_source(source):
        return
    current_frame = int(source.get(cv2.CAP_PROP_POS_FRAMES) or 0)
    frame_count = int(source.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    target_frame = int(round(current_frame + delta_sec * source_fps))
    if frame_count > 0:
        target_frame = min(frame_count - 1, target_frame)
    target_frame = max(0, target_frame)
    source.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    print(f"seek: {target_frame / max(source_fps, 0.1):.1f}s")


def seek_delta_from_key(key, args):
    if key == ord("d"):
        return args.skip_sec
    if key == ord("a"):
        return -args.skip_sec
    if key == ord("w"):
        return args.minute_skip_sec
    if key == ord("s"):
        return -args.minute_skip_sec
    return None


def skip_video_frames_for_preview(source, source_fps, preview_fps):
    stride = max(1, round(source_fps / max(preview_fps, 0.1)))
    for _ in range(stride - 1):
        if not source.grab():
            break


def sleep_to_preview_fps(loop_start, preview_fps):
    remaining = 1.0 / max(preview_fps, 0.1) - (time.monotonic() - loop_start)
    if remaining > 0:
        time.sleep(remaining)


def resolve_video_path(video_arg):
    path = Path(video_arg)
    if path.exists():
        return path
    if path.suffix:
        candidate = Path("images") / path
        return candidate if candidate.exists() else path

    videos = available_videos()
    if video_arg.isdigit():
        index = int(video_arg) - 1
        if 0 <= index < len(videos):
            return videos[index]
    for video in videos:
        if video.stem == video_arg:
            return video
    return Path("images") / f"{video_arg}.mp4"


def available_videos():
    images_dir = Path("images")
    if not images_dir.exists():
        return []
    return sorted(path for path in images_dir.iterdir() if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS)


def print_available_videos():
    videos = available_videos()
    if not videos:
        print("images 폴더에 선택 가능한 동영상이 없습니다.")
        return
    print("Available videos:")
    for idx, video in enumerate(videos, start=1):
        print(f"{idx}. {video.stem} ({video})")


def print_runtime_info(args, source_fps):
    print("ObjectAssist runtime")
    print("model:", OBJECT_MODEL_PATH)
    print(f"target_FPS: {args.fps:.1f}")
    print("preview:", args.preview)
    if args.preview:
        print(f"preview_FPS: {args.preview_fps:.1f}")
    if args.video:
        print(f"video_FPS: {source_fps:.1f}")


def speak_startup(speaker):
    if speaker is not None:
        speaker.speak("객체 보행 보조를 시작합니다.", key="startup", force=True)


def empty_metrics(target_fps):
    return {
        "latency_ms": None,
        "infer_ms": None,
        "infer_fps": 0.0,
        "target_fps": target_fps,
        "surface_fps": 0.0,
        "surface_target_fps": 0.0,
        "surface_status": "none",
    }


def metrics_from_times(start, done, target_fps):
    infer_ms = (done - start) * 1000.0
    return {
        "latency_ms": infer_ms,
        "infer_ms": infer_ms,
        "infer_fps": 1000.0 / max(infer_ms, 1e-6),
        "target_fps": target_fps,
        "surface_fps": 0.0,
        "surface_target_fps": 0.0,
        "surface_status": "none",
    }


def metrics_from_result(result, recent_fps, target_fps):
    return {
        "latency_ms": result.get("latency_ms"),
        "infer_ms": result.get("infer_ms"),
        "infer_fps": recent_fps,
        "target_fps": target_fps,
        "surface_fps": 0.0,
        "surface_target_fps": 0.0,
        "surface_status": "none",
    }


def preserve_surface_metrics(new_metrics, old_metrics):
    new_metrics["surface_fps"] = old_metrics.get("surface_fps", 0.0)
    new_metrics["surface_target_fps"] = old_metrics.get("surface_target_fps", 0.0)
    new_metrics["surface_status"] = old_metrics.get("surface_status", "none")
    return new_metrics


def update_recent_times(done_times, timestamp):
    done_times.append(timestamp)
    while done_times and timestamp - done_times[0] > 2.0:
        done_times.popleft()


def recent_fps_from_times(done_times):
    if len(done_times) < 2:
        return float(len(done_times))
    elapsed = max(done_times[-1] - done_times[0], 1e-6)
    return (len(done_times) - 1) / elapsed


def show_preview(frame, scale):
    display_frame = scaled_preview_frame(frame, scale)
    cv2.imshow(WINDOW_NAME, display_frame)


def scaled_preview_frame(frame, scale):
    scale = max(0.1, float(scale))
    if abs(scale - 1.0) < 0.01:
        return frame
    h, w = frame.shape[:2]
    return cv2.resize(frame, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_LINEAR)


def draw_preview(frame, detections, surface_detections, events, metrics, args, alert_log):
    """확인용 화면에 ROI, bbox, confidence, 상태 패널을 그립니다."""
    if args.surface_mask:
        draw_surface_masks(frame, surface_detections)
    if args.show_roi:
        draw_alert_roi(frame)
        draw_vehicle_roi(frame)
    draw_detections(frame, detections, args.overlay_text)
    if args.overlay_text:
        draw_status_overlay(frame, events, metrics, alert_log)


def draw_detections(frame, detections, show_score):
    h, w = frame.shape[:2]
    for idx, det in enumerate(detections[:MAX_PREVIEW_BOXES]):
        if det.box is None:
            continue
        x1, y1, x2, y2 = det.box
        p1 = (int(x1 * w), int(y1 * h))
        p2 = (int(x2 * w), int(y2 * h))
        color = color_for_label(det.label, idx)
        cv2.rectangle(frame, p1, p2, color, 2)
        if show_score:
            draw_score(frame, det.score, p1[0], p1[1] - 5, color)


def draw_surface_masks(frame, detections):
    if not detections:
        return
    overlay = frame.copy()
    mask_count = 0
    for idx, det in enumerate(detections):
        if det.mask is None:
            continue
        mask = det.mask.astype(bool)
        if mask.shape[:2] != frame.shape[:2]:
            mask = cv2.resize(mask.astype(np.uint8), (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
        color = np.array(surface_color_for_label(det.label, idx), dtype=np.uint8)
        overlay[mask] = (0.50 * overlay[mask] + 0.50 * color).astype(np.uint8)
        mask_count += 1
    if mask_count:
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, dst=frame)


def draw_status_overlay(frame, events, metrics, alert_log):
    alert_color = (0, 80, 255) if events else (80, 220, 80)
    lines = [
        f"obj={metrics.get('infer_fps', 0.0):.1f}/{metrics.get('target_fps', 0.0):.1f}",
        f"suf={metrics.get('surface_fps', 0.0):.1f}/{metrics.get('surface_target_fps', 0.0):.1f}",
        f"grd={metrics.get('surface_status', 'none')}",
        f"lat={format_ms(metrics.get('latency_ms'))}",
    ]
    log_items = alert_log.visible_items() if alert_log is not None else []
    if log_items:
        lines.extend(f"{item['time_text']} {item['key']}" for item in log_items)

    x, y = 8, 8
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.44
    thickness = 1
    line_h = 17
    width = max(cv2.getTextSize(line, font, scale, thickness)[0][0] for line in lines) + 16
    height = line_h * len(lines) + 10
    draw_panel(frame, x, y, min(width, frame.shape[1] - 16), height)
    for index, line in enumerate(lines):
        color = surface_status_color(metrics) if index == 2 else (235, 235, 235)
        if index >= 4:
            color = (0, 220, 255)
        cv2.putText(frame, line, (x + 8, y + 17 + index * line_h), font, scale, color, thickness, cv2.LINE_AA)


def surface_status_color(metrics):
    status = metrics.get("surface_status", "none")
    if status == "none" or ":" not in status:
        return (235, 235, 235)
    label = status.split(":", 1)[0]
    return surface_color_for_label(label, 0)


def draw_score(frame, score, x, y, color):
    text = f"{score:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.42
    thickness = 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    x = max(0, min(x, frame.shape[1] - tw - 2))
    if y - th < 0:
        y = min(frame.shape[0] - 2, y + th + 9)
    y = max(th + 2, min(y, frame.shape[0] - 2))
    cv2.putText(frame, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def draw_panel(frame, x, y, width, height):
    x2 = min(frame.shape[1], x + width)
    y2 = min(frame.shape[0], y + height)
    if x2 <= x or y2 <= y:
        return
    roi = frame[y:y2, x:x2]
    panel = np.zeros_like(roi)
    cv2.addWeighted(panel, 0.55, roi, 0.45, 0.0, roi)


def draw_alert_roi(frame):
    h, w = frame.shape[:2]
    cv2.polylines(frame, polygon_points(alert_roi_polygon(), w, h), isClosed=True, color=(0, 0, 255), thickness=2)


def draw_vehicle_roi(frame):
    h, w = frame.shape[:2]
    cv2.polylines(frame, polygon_points(vehicle_roi_polygon(), w, h), isClosed=True, color=(0, 140, 255), thickness=1)


def polygon_points(polygon, width, height):
    key = (width, height, tuple(polygon))
    points = POLYGON_POINT_CACHE.get(key)
    if points is None:
        points = np.array([[(int(x * width), int(y * height)) for x, y in polygon]], dtype=np.int32)
        POLYGON_POINT_CACHE[key] = points
    return points


def color_for_label(label, idx):
    if label in STATIC_OBSTACLE_LABELS:
        return (0, 255, 255)
    if label in MOVING_RISK_LABELS:
        return (0, 80, 255)
    if label in PERSON_LABELS:
        return (80, 220, 80)
    palette = [(255, 160, 40), (80, 160, 255), (220, 80, 220), (80, 255, 255)]
    return palette[idx % len(palette)]


def surface_color_for_label(label, idx):
    colors = {
        "sidewalk": (80, 220, 80),
        "braille_guide_blocks": (0, 220, 255),
        "roadway": (70, 70, 230),
        "alley": (60, 150, 255),
        "crosswalk": (255, 255, 255),
        "bike_lane": (255, 160, 40),
        "caution_zone": (0, 180, 255),
    }
    if label in colors:
        return colors[label]
    palette = [(80, 220, 80), (255, 160, 40), (80, 160, 255), (220, 80, 220)]
    return palette[sum(ord(ch) for ch in label) % len(palette)]


def format_ms(value):
    return "--ms" if value is None else f"{value:.0f}ms"


def format_alert_time(video_time_sec):
    if video_time_sec is None:
        return ""
    return f"[{format_video_time(video_time_sec)}] "


def format_video_time(seconds):
    seconds = max(0.0, float(seconds))
    minutes = int(seconds // 60)
    whole_seconds = int(seconds % 60)
    tenths = int((seconds - int(seconds)) * 10)
    return f"{minutes:02d}:{whole_seconds:02d}.{tenths}"


def event_cooldown_key(event):
    object_id = event.object_id or "global"
    return f"{event.key}:{object_id}"


def parse_args():
    parser = argparse.ArgumentParser(description="ObjectAssist camera/video runtime")
    parser.add_argument("--camera", type=int, default=0, help="USB 카메라 번호")
    parser.add_argument("--video", type=str, default=None, help="확인할 동영상 경로, images/ 파일명, 또는 목록 번호")
    parser.add_argument("--list-videos", action="store_true", help="images/ 폴더의 동영상 목록 출력")
    parser.add_argument("--start-sec", type=float, default=0.0, help="동영상 시작 위치")
    parser.add_argument("--skip-sec", type=float, default=5.0, help="동영상 preview에서 A/D로 이동할 초")
    parser.add_argument("--minute-skip-sec", type=float, default=60.0, help="동영상 preview에서 S/W로 이동할 초")
    parser.add_argument("--fps", type=float, default=None, help="목표 추론 FPS. 생략하면 config.py의 OBJECT_INFERENCE_FPS를 사용")
    parser.add_argument("--surface-fps", type=float, default=SURFACE_INFERENCE_FPS, help="지면 mask preview 목표 FPS")
    parser.add_argument("--surface-mask", dest="surface_mask", action="store_true", default=True, help="지면 segmentation mask와 지면 알림 사용")
    parser.add_argument("--no-surface-mask", dest="surface_mask", action="store_false", help="지면 segmentation mask와 지면 알림 끄기")
    parser.add_argument("--preview-fps", type=float, default=None, help="동영상 preview 표시 FPS. 생략하면 config.py의 PREVIEW_FPS를 사용")
    parser.add_argument("--window-scale", type=float, default=1.0, help="512x288 preview 확대 배율")
    parser.add_argument("--preview", action="store_true", default=SHOW_PREVIEW, help="OpenCV preview 창 표시")
    parser.add_argument("--no-preview", dest="preview", action="store_false")
    parser.add_argument("--no-speak", action="store_true", help="음성 출력 없이 터미널/화면 알림만 사용")
    parser.add_argument("--fast", action="store_true", help="동영상 확인 시 대기 없이 가능한 빠르게 처리")
    parser.add_argument("--overlay-text", dest="overlay_text", action="store_true", default=True, help="상태 패널과 confidence 표시")
    parser.add_argument("--no-overlay-text", dest="overlay_text", action="store_false", help="상태 패널과 confidence 숨김")
    parser.add_argument("--alert-log-sec", type=float, default=12.0, help="preview 알림 로그 유지 시간")
    parser.add_argument("--alert-log-size", type=int, default=5, help="preview 알림 로그 최대 개수")
    parser.add_argument("--no-roi", dest="show_roi", action="store_false", help="ROI 선 숨김")
    parser.set_defaults(show_roi=True)
    parser.add_argument("--terminal-status", action="store_true", default=True, help="터미널 상태 출력")
    parser.add_argument("--no-terminal-status", dest="terminal_status", action="store_false")
    parser.add_argument("--status-interval-sec", type=float, default=2.0, help="터미널 상태 출력 간격")
    parser.add_argument("--delay-ms", type=int, default=1, help="OpenCV 키 입력 대기 시간")
    args = parser.parse_args()
    if args.fps is None:
        args.fps = OBJECT_INFERENCE_FPS
    if args.preview_fps is None:
        args.preview_fps = PREVIEW_FPS
    args.surface_fps = min(max(args.surface_fps, 0.1), SURFACE_INFERENCE_FPS)
    return args


if __name__ == "__main__":
    run(parse_args())
