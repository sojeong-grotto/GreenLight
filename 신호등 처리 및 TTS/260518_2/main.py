import AI
import tts
import Operation
import multiprocessing as mp
from multiprocessing import shared_memory
import time
import camera
import os
import cv2
import numpy as np


if __name__ == '__main__':
    # deadlock, crash 방지를 위해 'spawn' 방식으로 설정
    mp.set_start_method('spawn', force=True)

    tts_worker = tts.TTSWorker('voice_list.json')

    res_queue = mp.Queue(maxsize=5)
    frame_ready = mp.Event()
    stop_event = mp.Event()

    # AI 프로세스와 공유할 프레임 버퍼 생성
    FRAME_SHAPE = (240, 320, 3)
    nbytes = int(np.prod(FRAME_SHAPE) * np.dtype(np.uint8).itemsize)
    shm = shared_memory.SharedMemory(create=True, size=nbytes)

    # AI 프로세스 시작 (shared memory 이름과 프레임 shape, event, queue 전달)
    AiProcess = mp.Process(target=AI.inference_process, args=(shm.name, FRAME_SHAPE, frame_ready, res_queue, stop_event))
    AiProcess.daemon = True
    AiProcess.start()

    # 카메라 초기화 및 캡처 루프는 메인 프로세스에서 실행 (Picamera2는 fork-safe하지 않음)
    Cam = camera.CameraInit()

    try:
        # TTSWorker가 초기화되고 음성 파일이 준비될 때까지 대기
        while not tts_worker._flag:
            time.sleep(0.1)

        print("[Main] capture loop 시작. Ctrl+C로 종료")

        while AiProcess.is_alive():
            frame = camera.CameraCapture(Cam, None)
            if frame is None:
                time.sleep(0.01)
                continue

            # shared memory에 직접 쓰기 (프레임이 크면 복사 비용이 있지만, 작은 프레임에서는 괜찮음)
            arr = np.ndarray(FRAME_SHAPE, dtype=np.uint8, buffer=shm.buf)

            # YOLO 모델이 512x288을 기대하지만, 카메라에서 320x240이 들어오면 모델에 맞게 조정 필요 (현재는 AI 프로세스에서 처리)
            if frame.shape != FRAME_SHAPE:
                resized = cv2.resize(frame, (FRAME_SHAPE[1], FRAME_SHAPE[0]))
            else:
                resized = frame
            arr[:] = resized[:]
            frame_ready.set()

            # AI 프로세스에서 결과가 오면 Operation.operator로 처리
            try:
                if not res_queue.empty():
                    data = res_queue.get_nowait()
                    # process using current frame
                    Operation.operator(resized, data, tts_worker)

                    # draw bounding box for quick visualization
                    x = data.get('x', 0)
                    y = data.get('y', 0)
                    w = data.get('w', 0)
                    h = data.get('h', 0)
                    # convert center xywh to left-top
                    lx = int(x - (w / 2))
                    ty = int(y - (h / 2))
                    rx = int(lx + w)
                    by = int(ty + h)
                    cv2.rectangle(resized, (lx, ty), (rx, by), (0, 255, 0), 2)
                    cv2.imshow('Camera', resized)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            except Exception:
                pass

            time.sleep(0.01)

    except KeyboardInterrupt:
        print('\n[Main] 프로그램을 종료합니다...')
    finally:
        # stop_event.set()는 AI 프로세스가 종료될 때까지 기다리는 역할도 함
        # AI 프로세스가 종료될 때까지 최대 2초 기다림
        stop_event.set()
        AiProcess.join(timeout=2)

        try:
            shm.close()
            shm.unlink()
        except Exception:
            pass

        try:
            Cam.stop()
        except Exception:
            pass

        tts_worker.stop()
        cv2.destroyAllWindows()
