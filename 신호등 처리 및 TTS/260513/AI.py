import time
import os

def inference_process(input_data, result_queue):
    # 무거운 AI 추론 — 별도 프로세스

    # # 1. 현재 실행 중인 파일(AI.py 등)의 폴더 경로를 구합니다.
    current_dir = os.path.dirname(os.path.abspath(__file__))

    try:

        while   True:

            image_path = "ex_traffic_light.png"
            # 여기에 AI 처리된 데이터의 이름이 들어가야함. 단, 이미지 파일은 해당 파일과 같은 경로에 있어야 함
            
            
            
            # 2. 파일 이름만 들어와도 현재 폴더 경로와 합쳐서 전체 경로를 만듭니다.
            # 만약 이미 전체 경로가 들어온다면 알아서 처리해줍니다.
            filepath = os.path.join(current_dir, image_path)

            # 밑에 frame과 x, y, w, h에 각각 맞는 처리된 데이터가 들어가야함.
            Path = filepath
            x, y, w, h = 50, 50, 10, 10

            Data = {"Path" : Path, "x" : x, "y" : y, "w" : w, "h" : h}
            try:
                    
                if result_queue.full():
                    try:
                        result_queue.get_nowait() # 가장 오래된 경로 데이터 하나 삭제
                    except:
                        pass

                # 2. 큐에 데이터 넣기 (Put)
                result_queue.put(Data)
            except Exception as e:
                # 큐 관련 에러 발생 시 출력 (보통 메인 프로세스 종료 시 발생)
                print(f"Queue Error: {e}")
                break

            time.sleep(1)

    except KeyboardInterrupt:
        # 사용자가 프로그램을 끌 때(Ctrl+C) 발생하는 에러 로그를 숨깁니다.
        print("keyboard Interrupt로 프로그램이 종료됩니다.")
        pass
    except EOFError:
        # 프로세스 간 통신이 끊겼을 때 조용히 종료합니다.
        print("프로세스 간 통신이 끊겨 종료됩니다.")
        pass