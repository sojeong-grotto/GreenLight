import time
import os
import cv2
import numpy as np
import camera
from ultralytics import YOLO


TARGET_W, TARGET_H = 512, 288  # 모델 입력 사이즈 및 캔버스 크기

def inference_process(input_data, result_queue):
    # 무거운 AI 추론 — 별도 프로세스

    global camera

    Cam = camera.CameraInit()

    # # 1. 현재 실행 중인 파일(AI.py 등)의 폴더 경로를 구합니다.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 전체 경로 생성
    target_path = os.path.join(current_dir, "captured_images")
    print(target_path)

    try:
        YoloModelPath = os.path.join(current_dir, "PedAssist_crosswalk_signal_yolo26n-seg.pt")
        print(YoloModelPath)
        model = YOLO(YoloModelPath)

        while   True:
            
            max_area = 0
            largest_box = None
            curr_x, curr_y, curr_w, curr_h = 0, 0, 0, 0
            
            camera.CameraCapture(Cam, target_path)
            image_path = f"img_{camera.CamCnt}.jpg"
            camera.CamCnt += 1 # CamCnt 위치를 image_path 밑에 해야 위에서 생성되는 경로 값이 제대로 생성됨. 위치 중요!!
            # 위치 중요 : 여기에 AI 처리된 데이터의 이름이 들어가야함. 단, 이미지 파일은 해당 파일과 같은 경로에 있어야 함
            
            # 2. 파일 이름만 들어와도 현재 폴더 경로와 합쳐서 전체 경로를 만듭니다.
            # 만약 이미 전체 경로가 들어온다면 알아서 처리해줍니다.
            
            filepath = os.path.join(current_dir, camera.IMAGE_FOLDER_NAME)
            filepath = os.path.join(filepath, image_path)
            
            results = model(filepath) # , stream=True 필요하려나?
        
            for r in results:
                # r.boxes에는 검출된 모든 박스 정보가 담겨 있음
                for box in r.boxes:
                    # xywh: 중심점(x, y)과 너비(w), 높이(h) / xyxy: 좌상단, 우하단 좌표
                    # 여기서는 요구하신 대로 xywh를 사용합니다. (.cpu().numpy()는 데이터 처리를 위함)
                    b = box.xywh[0].cpu().numpy() 
                    x, y, w, h = b
                    
                    area = w * h
                    if area > max_area:
                        max_area = area
                        largest_box = (x, y, w, h)

                # 시각화된 프레임 가져오기
                annotated_frame = r.plot()

            # 가장 큰 박스가 있을 경우 좌표 출력
            if largest_box:
                curr_x, curr_y, curr_w, curr_h = largest_box
                # f-string을 사용하여 소수점 둘째자리까지 출력
                print(f"Largest Box -> Center X: {curr_x:.2f}, Center Y: {curr_y:.2f}, Width: {curr_w:.2f}, Height: {curr_h:.2f}")

            cv2.imshow("YOLO Inference", annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break



            # 밑에 frame과 x, y, w, h에 각각 맞는 처리된 데이터가 들어가야함.
            Left_x = curr_x - (curr_w / 2)
            Top_y = curr_y - (curr_h / 2)
            x, y, w, h = Left_x, Top_y, curr_w, curr_h

            Data = {"Path" : filepath, "x" : x, "y" : y, "w" : w, "h" : h}
            
            try:
                # 3. 큐에 데이터 넣기 (Put)
                result_queue.put(Data)
            except Exception as e:
                # 큐 관련 에러 발생 시 출력 (보통 메인 프로세스 종료 시 발생)
                print(f"Queue Error: {e}")
                break
            
            
            
            time.sleep(0.1)

    except KeyboardInterrupt:
        # 사용자가 프로그램을 끌 때(Ctrl+C) 발생하는 에러 로그를 숨깁니다.
        print("keyboard Interrupt로 프로그램이 종료됩니다.")
        pass
    except EOFError:
        # 프로세스 간 통신이 끊겼을 때 조용히 종료합니다.
        print("프로세스 간 통신이 끊겨 종료됩니다.")
        pass

