import time
import os
import cv2
import numpy as np
from ultralytics import YOLO
from multiprocessing import shared_memory


pt_model_name = "PedAssist_crosswalk_signal_yolo26n-seg.pt"
tflite_model_name = "yolo26n-seg_full_integer_quant.tflite"
def inference_process(shared_name, shape, frame_ready, result_queue, stop_event):
    # 공유 메모리 버퍼에서 프레임을 읽어 YOLO 모델로 추론하는 프로세스
    # Args:
    #     shared_name (str): 공유 메모리 블록 이름
    #     shape (tuple): 프레임의 형태 e.g. (H, W, 3)
    #     frame_ready (multiprocessing.Event): 새 프레임이 준비되면 producer가 설정
    #     result_queue (multiprocessing.Queue): 감지 메타데이터를 main으로 보내는 큐
    #     stop_event (multiprocessing.Event): 설정되면 프로세스가 종료되어야 함

    current_dir = os.path.dirname(os.path.abspath(__file__))

    # TFLite 모델이 있지만, 현재는 .pt 모델이 존재할 때만 사용하도록 설정. TFLite 경로 처리 및 로드 로직은 추후 구현 예정.
    pt_model = os.path.join(current_dir, pt_model_name)
    tflite_model = os.path.join(current_dir, tflite_model_name)

    model = None
    use_ultralytics = False
    if os.path.exists(pt_model):
        try:
            model = YOLO(pt_model)
            use_ultralytics = True
            print(f"[AI] Loaded PyTorch model: {pt_model}")
        except Exception as e:
            print(f"[AI] Failed to load .pt model: {e}")

    # main process에서 생성된 공유 메모리 블록에 연결
    shm = shared_memory.SharedMemory(name=shared_name)
    frame_shape = tuple(shape)

    try:
        while not stop_event.is_set():
            # 새로운 프레임이 없으면 루프를 계속 돌면서 stop_event 체크
            # timeout을 주어 무한 대기 방지 (예: 1초)
            if not frame_ready.wait(timeout=1.0):
                continue

            # 공유 메모리에서 프레임을 읽어 numpy 배열로 변환
            arr = np.ndarray(frame_shape, dtype=np.uint8, buffer=shm.buf)
            frame = arr.copy()

            # frame_ready 이벤트를 초기화하여 다음 프레임을 기다림
            frame_ready.clear()

            if use_ultralytics and model is not None:
                try:
                    results = model(frame)

                    max_area = 0
                    largest_box = None
                    annotated_frame = None
                    for r in results:
                        for box in r.boxes:
                            b = box.xywh[0].cpu().numpy()
                            x, y, w, h = b
                            area = w * h
                            if area > max_area:
                                max_area = area
                                largest_box = (x, y, w, h)

                        annotated_frame = r.plot()

                    if largest_box:
                        curr_x, curr_y, curr_w, curr_h = largest_box
                        print(f"Largest Box -> Center X: {curr_x:.2f}, Center Y: {curr_y:.2f}, Width: {curr_w:.2f}, Height: {curr_h:.2f}")
                    else:
                        curr_x = curr_y = curr_w = curr_h = 0

                    # [참고] 멀티프로세싱 환경에서 GUI(imshow)를 사용할 때 프레임이 안 깨지도록 예외 처리 검토 필요
                    if annotated_frame is not None:
                        cv2.imshow("YOLO Inference", annotated_frame)
            
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                    curr_x = curr_x - (curr_w / 2)
                    curr_y = curr_y - (curr_h / 2)
                    Data = {"x": float(curr_x), "y": float(curr_y), "w": float(curr_w), "h": float(curr_h)}
                    
                    try:
                        result_queue.put(Data, timeout=0.5)
                    except Exception as e:
                        print(f"[AI] result_queue.put failed: {e}")

                except Exception as e:
                    print(f"[AI] Inference error: {e}")
                    time.sleep(0.1)
            else:
                # 모델이 없거나 로드 실패 시, 빈 데이터 전송
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("keyboard Interrupt로 프로그램이 종료됩니다.")
        pass
    except EOFError:
        print("프로세스 간 통신이 끊겨 종료됩니다.")
        pass
    finally:
        try:
            shm.close()
        except:
            pass