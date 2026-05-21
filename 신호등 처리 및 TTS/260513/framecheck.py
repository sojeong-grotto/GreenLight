import cv2
import numpy as np


MINIMUM_COUNT_FOR_CHECK_COLOR = 3 # 3프레임으로 임의 설정
# 이 값 이상은 count가 되어야 State가 변했다고 인식함.
# 즉, 이 값은 실제 Frame이 얼마가 나오는지 확인 필요.
# default value : 3 ==> 1초에 10프레임은 나온다고 예상하고 설정. 최소 0.3초는 같은 색이 보여야 해당 state로 변경하겟다는 의미
# 중요 : 불이 점멸하는 시간보다는 짧게 잡아야 점멸하는걸 인식할 수 있기에 검은색 유지 시간 보다 짧아야 함
# 중요 : 불이 점멸하는 시간 : 0.5초 초록 --> 0.5초 검정 --> 0.5초 초록 으로 반복한다고 가정(재미나이 님께서 알려주심)

MINIMUM_COUNT_FOR_INITIALIZATION = 50 # 50프레임으로 임의 설정
# 1초에 10프레임이 나온다고 생각하고, 5초는 신호등이 연속으로 안보여야 없다고 판단하도록 설정

MINIMUM_COUNT_FOR_CHECK_BLINK = 2 # 필요 없을듯

CurrentState = "NoSignal"
PreviousColor, RecognizedColor = None, None
RecognizeColorCnt, ChangeStateCnt = 0, 0
CheckBlinkFlag = False


# 이전상태와 현재상태를 비교

def isBoxInFrame(Frame, x, y, w, h): # Frame : 이미지, x, y : x, y 축 박스의 좌표, w : x축 박스 길이, h : y축 박스 길이
    
    Height, Width, Channels = Frame.shape
    InputCoordinateFlag = False

    if (w <= 0) or (h <= 0):
        return False

    if x <= Width and (x+w) <= Width : # 가로 크기 확인
        if y <= Height and (y+h) <= Height : # 세로 크기 확인
            InputCoordinateFlag = True
        else: # ..필요없는 else 구문...(가독성을 위해 추가)
            InputCoordinateFlag = False
    else:
        InputCoordinateFlag = False

    if InputCoordinateFlag == True:
        return True
    else:
        return False
    

def determineColor(Frame, x, y, w, h):
    # 반환 값 : None, Black, Red, Green
    
    if  isBoxInFrame(Frame, x, y, w, h) == False:
        print("Frame 내에 박스가 없습니다.")
        return None
    
    RoiCoordinates = Frame[y:y+h, x:x+w] # 네모 상자가 그려진 부분의 좌표(y축 시점, y축 종점, x축 시점, x축 종점)
    HsvRoi = cv2.cvtColor(RoiCoordinates, cv2.COLOR_BGR2HSV) # BGR Data ==> HSV 데이터로 변경

    # 빨간색
    lower_red1, upper_red1 = np.array([0, 100, 100]), np.array([10, 255, 255])
    lower_red2, upper_red2 = np.array([160, 100, 100]), np.array([180, 255, 255])
    # 초록색
    lower_green, upper_green = np.array([40, 100, 100]), np.array([90, 255, 255])

    # 3. 마스크 생성
    MaskRed = cv2.add(cv2.inRange(HsvRoi, lower_red1, upper_red1),
                        cv2.inRange(HsvRoi, lower_red2, upper_red2))
    MaskGreen = cv2.inRange(HsvRoi, lower_green, upper_green)

    # 4. 픽셀 갯수 Count
    RedCount = cv2.countNonZero(MaskRed)
    GreenCount = cv2.countNonZero(MaskGreen)
    
    # 5. 선택한 영역에서의 색상 Pixel 갯수 Count
    TotalPixel = w * h
    RedRatio = (RedCount / TotalPixel) * 100
    GreenRatio = (GreenCount / TotalPixel) * 100

    print(f"RedRatio = {RedRatio}, GreenRatio = {GreenRatio}")

    if  (RedRatio < 0.2) and (GreenRatio < 0.2): # 전체 픽셀 중 초록, 빨강이 20%가 안될 때 --> 불이 둘다 꺼진걸로 확인
        return("Black")
    else:
        if  RedRatio >= GreenRatio: # 빨강색  판정(두개의 비율이 같더라도 안전을 위해 빨강으로 판단)
            return("Red")
        else:
            return("Green")
    

def determineState(Color):
    # Color 값 : None, Black, Red, Green
    # 반환 값 : NoSignal(신호등 없을 때), LightingState(신호가 켜져 있을 때) BlinkingState(점멸), BrokenState(고장)
    global RecognizeColorCnt, PreviousColor, CurrentState, CheckBlinkFlag, RecognizedColor, ChangeStateCnt
    
    if  not (PreviousColor == Color):
        RecognizeColorCnt = 0
        ChangeStateCnt = 0
        PreviousColor = Color
        State = CurrentState
    else:
        if RecognizeColorCnt >= MINIMUM_COUNT_FOR_CHECK_COLOR:
            RecognizeColorCnt = 100 # RecognizeColorCnt OverFlow 방지...할 필요가 있나? --> For Fix
            
            if      Color == None:
                if  ChangeStateCnt >= MINIMUM_COUNT_FOR_INITIALIZATION: # 5초 이상 시
                    RecognizedColor = None
                    State = "NoSignal"
                else:
                    State = CurrentState
                    ChangeStateCnt += 1

            elif    Color == "Black":

                if  CheckBlinkFlag == True:
                     State = "BlinkingState"
                else:
                    if  RecognizedColor == "Green": # 이전 색상이 초록색 이었다가 검은색이 됬을 때
                        CheckBlinkFlag = True
                        State = "BlinkingState"
                    else:
                        State = "BrokenState"

                RecognizedColor = "Black"
                
            elif    Color == "Red":
                    State = "LightingState"
                    RecognizedColor = "Red"
            
            elif    Color == "Green":
                if  CheckBlinkFlag == True:
                     State = "BlinkingState"
                else:
                    if  RecognizedColor == "Black": # 이전 색상이 초록색 이었다가 검은색이 됬을 때
                        CheckBlinkFlag = True
                        State = "BlinkingState"
                    else:
                        State = "LightingState"

                RecognizedColor = "Green"
            else:
                print("색상 변화에서 Error가 발생하였습니다.")
                State = "NoSignal" # Error임
                RecognizedColor = None # Error임 
        else:
            RecognizeColorCnt += 1
            State = CurrentState
    return State