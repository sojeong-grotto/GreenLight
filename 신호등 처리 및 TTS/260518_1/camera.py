import os
import cv2
import numpy as np
import time
from picamera2 import Picamera2

IMAGE_FOLDER_NAME = "captured_images"
CamCnt = 1

# 최종 목적지 사이즈 (YOLO 입력 사이즈)
TARGET_W, TARGET_H = 512, 288

def CameraInit():
    Cam = Picamera2()
    # BGR888로 시도하지만, 시스템에 따라 YUYV로 설정될 수 있음
    config = Cam.create_preview_configuration(main={'format': 'BGR888', 'size': (320, 240)})
    Cam.configure(config)
    Cam.start()
    time.sleep(2) 
    return Cam

def CameraCapture(Cam, save_dir):
    global CamCnt

    # 1. 프레임 가져오기
    raw_frame = Cam.capture_array()
    
    if raw_frame is None:
        return None

    # 2. 채널 수 확인 및 BGR 변환 (중요!)
    # shape가 (H, W, 2)이면 YUYV 포맷이므로 BGR로 변환해줘야 함
    if len(raw_frame.shape) == 3 and raw_frame.shape[2] == 2:
        frame = cv2.cvtColor(raw_frame, cv2.COLOR_YUV2BGR_YUYV)
    elif len(raw_frame.shape) == 3 and raw_frame.shape[2] == 3:
        # 이미 3채널(RGB/BGR)인 경우 (설정이 먹혔을 때)
        frame = raw_frame
    else:
        # 그 외 예외 케이스
        frame = raw_frame

    # # 3. 이미지 리사이즈 및 패딩 (Letterbox 처리)
    # h, w = frame.shape[:2]
    # scale = min(TARGET_W / w, TARGET_H / h)
    # nw, nh = int(w * scale), int(h * scale)
    
    # resized_img = cv2.resize(frame, (nw, nh))
    
    # # 3채널 검은색 캔버스(512x288, 3) 생성
    # canvas = np.zeros((TARGET_H, TARGET_W, 3), dtype=np.uint8)
    
    # x_offset = (TARGET_W - nw) // 2
    # y_offset = (TARGET_H - nh) // 2
    
    # # 이제resized_img도 3채널이므로 캔버스(3채널)에 복사가 가능합니다!
    # canvas[y_offset:y_offset+nh, x_offset:x_offset+nw] = resized_img

    # 4. 파일 저장
    filename = os.path.join(save_dir, f'img_{CamCnt}.jpg')
    cv2.imwrite(filename, frame)
    # cv2.imwrite(filename, canvas)
    
    # return canvas