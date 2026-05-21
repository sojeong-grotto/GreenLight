import time
import os
import cv2
import numpy as np
import camera
from ultralytics import YOLO
import tflite_runtime.interpreter as tflite

TARGET_W, TARGET_H = 512, 288  # 모델 입력 사이즈 및 캔버스 크기
ModelName = "yolo26n-seg_full_integer_quant.tflite" # 변수명 직관적으로 변경

def inference_process(input_data, result_queue):
    # 무거운 AI 추론 — 별도 프로세스
    global camera

    Cam = camera.CameraInit()

    # 1. 현재 실행 중인 파일(AI.py)의 폴더 경로를 구합니다.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # [수정] 모델의 전체 절대 경로를 먼저 생성합니다.
    TFLiteModelPath = os.path.join(current_dir, ModelName)
    print(f"Model Path: {TFLiteModelPath}")

    # [수정] 상대 경로 대신 절대 경로(TFLiteModelPath)를 전달합니다.
    interpreter = tflite.Interpreter(model_path = TFLiteModelPath)
    interpreter.allocate_tensors()

    # 전체 경로 생성 (이미지 저장용)
    target_path = os.path.join(current_dir, "captured_images")
    print(f"Target Path: {target_path}")

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_index = input_details[0]["index"]
    output_index = output_details[0]["index"]
    input_dtype = input_details[0]["dtype"]
    output_dtype = output_details[0]["dtype"]

    input_scale, input_zero = interpreter.get_input_details()[0]["quantization"]
    output_scale, output_zero = interpreter.get_output_details()[0]["quantization"]

    height = input_details[0]["shape"][1]
    width = input_details[0]["shape"][2]

    try:
        # [수정] 이미 위에서 TFLiteModelPath를 만들었으므로 중복 코드 제거 및 로드
        model = YOLO(TFLiteModelPath)

        while True:
            max_area = 0
            largest_box = None
            curr_x, curr_y, curr_w, curr_h = 0, 0, 0, 0
            
            camera.CameraCapture(Cam, target_path)
            image_path = f"img_{camera.CamCnt}.jpg"
            camera.CamCnt += 1 
            
            filepath = os.path.join(current_dir, camera.IMAGE_FOLDER_NAME)
            filepath = os.path.join(filepath, image_path)
            
            results = model(filepath) 
        
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

            # [참고] 멀티프로세싱 환경에서 GUI(imshow)를 사용할 때 프레임이 안 깨지도록 예외 처리 검토 필요
            if annotated_frame is not None:
                cv2.imshow("YOLO Inference", annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            Left_x = curr_x - (curr_w / 2)
            Top_y = curr_y - (curr_h / 2)
            x, y, w, h = Left_x, Top_y, curr_w, curr_h

            Data = {"Path" : filepath, "x" : x, "y" : y, "w" : w, "h" : h}
            
            try:
                result_queue.put(Data)
            except Exception as e:
                print(f"Queue Error: {e}")
                break
            
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("keyboard Interrupt로 프로그램이 종료됩니다.")
        pass
    except EOFError:
        print("프로세스 간 통신이 끊겨 종료됩니다.")
        pass