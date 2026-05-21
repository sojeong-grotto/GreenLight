import framecheck
import tts
import time
import cv2
import os

PreviousColor, PreviousState = None, None
# operator now processes a frame and metadata (not file paths)
def operator(frame, data, tts_worker):
    # Color 값 : None, Black, Red, Green
    # 반환 값 : NoSignal(신호등 없을 때), LightingState(신호가 켜져 있을 때) BlinkingState(점멸), BrokenState(고장)
    global PreviousColor, PreviousState, FlagForPlayOnce

    if data is None:
        return None

    try:
        x, y, w, h = data.get("x"), data.get("y"), data.get("w"), data.get("h")

        if frame is None or x is None or y is None or w is None or h is None:
            print("Queue Data Error\n잘못된 Data가 입력되었습니다.")
            return None

        Color = framecheck.determineColor(frame, x, y, w, h)
        State = framecheck.determineState(Color)

        if (not (PreviousColor == Color)) or (not (PreviousState == State)):
            FlagForPlayOnce = False

        TTS_Status = None
        if State == "NoSignal":
            print(f"State는 {State}입니다.")
        elif State == "LightingState":
            if Color == "Red":
                TTS_Status = tts.dicSpeechText["빨간 신호"]
                print(f"State는 {State}입니다.")
            else:
                TTS_Status = tts.dicSpeechText["초록 신호"]
        elif State == "BlinkingState":
            print(f"State는 {State}입니다.")
        elif State == "BrokenState":
            print(f"State는 {State}입니다.")
        else:
            print(f"Error => State는 {State}입니다.")

        if FlagForPlayOnce == False and TTS_Status is not None:
            tts_worker._status = TTS_Status

        PreviousColor = Color
        PreviousState = State

    except Exception as e:
        print(f"Operator error: {e}")
        return None

    except: # 큐가 비어있으면 그냥 패스
        pass

