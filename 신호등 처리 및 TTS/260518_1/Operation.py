import framecheck
import tts
import time
import cv2
import os

PreviousColor, PreviousState = None, None
FlagForPlayOnce = False

# 프로세스로 동작할 2번째 친구임.
def operator(ResultQueue, tts_worker):
    # Color 값 : None, Black, Red, Green
    # 반환 값 : NoSignal(신호등 없을 때), LightingState(신호가 켜져 있을 때) BlinkingState(점멸), BrokenState(고장)
    global PreviousColor, PreviousState
    
    TTS_Status = None # 반복해서 0이 되어도 상관 없음



    try:
        try:
            # 1초 정도의 타임아웃을 주면 더 안전합니다.
            QData = ResultQueue.get(timeout=1) 
        except:
            # 큐가 비었거나 대기 중일 때 통과
            return None

        Frame = cv2.imread(QData["Path"])
        x, y, w, h = QData["x"], QData["y"], QData["w"], QData["h"]

        if os.path.exists(QData["Path"]):
            os.remove(QData["Path"])
            

        if (x is None) or (y is None) or (w is None) or (h is None) or (Frame is None):
            print("Queue Data Error\n잘못된 Data가 입력되었습니다.")
            QData = None
            return None

        Color = framecheck.determineColor(Frame, x, y, w, h)
        State = framecheck.determineState(Color)

        if (not (PreviousColor == Color)) or (not (PreviousState == State)):
            FlagForPlayOnce = False
        
        if      State == "NoSignal":
            print(f"State는 {State}입니다.")

        elif    State == "LightingState":
            if  Color == "Red": 

                # 예시
                TTS_Status = tts.dicSpeechText["빨간 신호"]
                print(f"State는 {State}입니다.")
                # 예시
            else:
                TTS_Status = tts.dicSpeechText["초록 신호"]

        elif    State == "BlinkingState":
            print(f"State는 {State}입니다.")
        
        elif    State == "BrokenState":
            print(f"State는 {State}입니다.")

        else:
            print(f"Error => State는 {State}입니다.")


        if FlagForPlayOnce == False:    
            tts_worker._status = TTS_Status
        PreviousColor = Color
        PreviousState = State

    except: # 큐가 비어있으면 그냥 패스
        pass
