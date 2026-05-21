import time

import cv2

from config import CAMERA_CROP_TO_TARGET_ASPECT, CAMERA_FPS, CAMERA_HEIGHT, CAMERA_WIDTH


class Camera:
    def __init__(self, width=CAMERA_WIDTH, height=CAMERA_HEIGHT, fps=CAMERA_FPS, camera_index=0):
        self.width = width
        self.height = height
        self.fps = fps
        self.camera_index = camera_index
        self.picam2 = None
        self.cap = None
        self.last_read_time = 0.0

    def start(self):
        picamera_errors = []
        try:
            from picamera2 import Picamera2

            for use_controls in (True, False):
                try:
                    self.picam2 = Picamera2(self.camera_index)
                    kwargs = {"main": {"size": (self.width, self.height), "format": "RGB888"}}
                    if use_controls:
                        kwargs["controls"] = {"FrameDurationLimits": self._frame_duration_limits()}
                    config = self.picam2.create_preview_configuration(**kwargs)
                    self.picam2.configure(config)
                    self.picam2.start()
                    print(f"Picamera2 camera opened. controls={use_controls}")
                    return self
                except Exception as exc:
                    picamera_errors.append(f"controls={use_controls}: {type(exc).__name__}: {exc}")
                    if self.picam2 is not None:
                        try:
                            self.picam2.close()
                        except Exception:
                            pass
                    self.picam2 = None
        except Exception as exc:
            picamera_errors.append(f"import/create: {type(exc).__name__}: {exc}")

        self.cap = cv2.VideoCapture(self.camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        if not self.cap.isOpened():
            detail = "\n".join(f"- {err}" for err in picamera_errors) or "- Picamera2 시도 내역 없음"
            raise RuntimeError(
                "카메라를 열 수 없습니다. Picamera2 또는 USB 카메라 연결을 확인하세요.\n"
                f"Picamera2 실패 내역:\n{detail}"
            )
        print(f"OpenCV camera opened. index={self.camera_index}")
        return self

    def read(self):
        self._throttle()
        if self.picam2 is not None:
            frame_rgb = self.picam2.capture_array()
            if frame_rgb.ndim == 3 and frame_rgb.shape[2] == 2:
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_YUV2BGR_YUYV)
            elif frame_rgb.ndim == 3 and frame_rgb.shape[2] == 4:
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGBA2BGR)
            elif frame_rgb.ndim == 3 and frame_rgb.shape[2] == 3:
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            else:
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_GRAY2BGR)
            return crop_to_aspect_and_resize(frame_bgr, self.width, self.height)

        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("카메라 프레임을 읽지 못했습니다.")
        return crop_to_aspect_and_resize(frame, self.width, self.height)

    def stop(self):
        if self.picam2 is not None:
            self.picam2.stop()
            self.picam2 = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def _frame_duration_limits(self):
        frame_us = int(1_000_000 / max(self.fps, 1))
        return frame_us, frame_us

    def _throttle(self):
        interval = 1.0 / max(self.fps, 1)
        now = time.monotonic()
        wait = interval - (now - self.last_read_time)
        if wait > 0:
            time.sleep(wait)
        self.last_read_time = time.monotonic()


def crop_to_aspect_and_resize(frame_bgr, width=CAMERA_WIDTH, height=CAMERA_HEIGHT):
    # Training images were prepared as 16:9 frames. If the webcam delivers
    # another aspect ratio, crop first and then resize so objects are not
    # horizontally or vertically stretched.
    if frame_bgr is None:
        return frame_bgr

    src_h, src_w = frame_bgr.shape[:2]
    if src_w <= 0 or src_h <= 0:
        return frame_bgr

    if CAMERA_CROP_TO_TARGET_ASPECT:
        target_aspect = width / height
        src_aspect = src_w / src_h

        if src_aspect > target_aspect:
            crop_w = int(round(src_h * target_aspect))
            x1 = max(0, (src_w - crop_w) // 2)
            frame_bgr = frame_bgr[:, x1 : x1 + crop_w]
        elif src_aspect < target_aspect:
            crop_h = int(round(src_w / target_aspect))
            y1 = max(0, (src_h - crop_h) // 2)
            frame_bgr = frame_bgr[y1 : y1 + crop_h, :]

    if frame_bgr.shape[1] == width and frame_bgr.shape[0] == height:
        return frame_bgr
    return cv2.resize(frame_bgr, (width, height), interpolation=cv2.INTER_LINEAR)
