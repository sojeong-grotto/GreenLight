import cv2
import numpy as np

def CheckCaptureSize(Height, Width, x, y, w, h): # 이미지 capture 할 부분의 size가 원시 이미지 크기보다 작은지 여부 확인하여 반환
    
    InputCoordinateFlag = False

    if x <= Width and (x+w) <= Width : # 가로 크기 확인
        if y <= Height and (y+h) <= Height : # 세로 크기 확인
            InputCoordinateFlag = True
        else: # ..필요없는 else 구문...(가독성을 위해 추가)
            InputCoordinateFlag = False
    else:
        InputCoordinateFlag = False

    return InputCoordinateFlag

def CaptureImage(Image, x, y, w, h): # 이미지 중 원하는 영역만을 capture하여 반환
    
    Height, Width, Channels = Image.shape # 이미지의 shape(세로, 가로, 채널의 수)를 각 변수에 저장

    print(f"이미지 높이 (Height): {Height} px")
    print(f"이미지 너비 (Width): {Width} px")

    InputCoordinateFlag = CheckCaptureSize(Height, Width, x, y, w, h)

    RoiCoordinates = None # 이미지 중 capture 할 부분 선택을 위한 변수

    if InputCoordinateFlag == False:
        print("Coordinate Error")
        RoiCoordinates = None
    else:
        RoiCoordinates = Image[y:y+h, x:x+w]
        
    return RoiCoordinates
    
def DetermineColor(path, x, y, w, h):

    Image = cv2.imread(path) # path에서 이미지 불러와 Image 변수에 저장

    RoiCoordinates = CaptureImage(Image, x, y, w, h)
    
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

    print(RedRatio, GreenRatio)

    if RedRatio > GreenRatio:
        Color = "RED"
        print("RED")
    elif GreenRatio > RedRatio:
        Color = "GREEN"
        print("GREEN")
    else:
        Color = "NONE"
        print("NONE")

    cv2.rectangle(Image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    cv2.imshow('CaptureImage', Image)    

    return Color
    

Color = DetermineColor("/home/willtek/work/examples/05_Object_Detection_Based_On-Device_AI/detection_result.jpg",150,190,35,35)

cv2.waitKey(0)