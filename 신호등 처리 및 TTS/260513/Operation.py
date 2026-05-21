import framecheck
import tts
import time
import cv2

# 프로세스로 동작할 2번째 친구임.
def operator(ResultQueue):
    # Color 값 : None, Black, Red, Green
    # 반환 값 : NoSignal(신호등 없을 때), LightingState(신호가 켜져 있을 때) BlinkingState(점멸), BrokenState(고장)
    try:
        try:
            # 1초 정도의 타임아웃을 주면 더 안전합니다.
            QData = ResultQueue.get(timeout=1) 
        except:
            # 큐가 비었거나 대기 중일 때 통과
            return None

        Frame = cv2.imread(QData["Path"])
        x, y, w, h = QData["x"], QData["y"], QData["w"], QData["h"]

        if (x is None) or (y is None) or (w is None) or (h is None) or (Frame is None):
            print("Queue Data Error\n잘못된 Data가 입력되었습니다.")
            QData = None
            return None

        Color = framecheck.determineColor(Frame, x, y, w, h)
        State = framecheck.determineState(Color)

        if      State == "NoSignal":
            print(f"State는 {State}입니다.")

        elif    State == "LightingState":
            print(f"State는 {State}입니다.")

        elif    State == "BlinkingState":
            print(f"State는 {State}입니다.")
        
        elif    State == "BrokenState":
            print(f"State는 {State}입니다.")

        else:
            print(f"Error => State는 {State}입니다.")
    except: # 큐가 비어있으면 그냥 패스
        pass

