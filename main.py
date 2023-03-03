import cv2, os
from cvzone.HandTrackingModule import HandDetector

width, height = 1280, 720
folderPath = "Presentation"
imgNo=0
hs, ws = int(120*1), int(213*1)
gestureThreshold=550
buttonPressed = False
buttonCounter = 0
buttonDelay = 15

#camera setup
cap = cv2.VideoCapture(0)
cap.set(3,width)
cap.set(4,height)

#get the list of presentation images
pathImages = sorted(os.listdir(folderPath), key=len)
# print(pathImages)

#Hand Detector
detector = HandDetector(detectionCon=0.8, maxHands=1)
while True:
    #Import Images
    success, img=cap.read()
    img = cv2.flip(img,1)
    pathFullImage = os.path.join(folderPath,pathImages[imgNo])
    currentImg = cv2.imread(pathFullImage)

    hands, img = detector.findHands(img, flipType=False) #use ', flipType=False' to flip the camera.
    cv2.line(img,(0, gestureThreshold),(width,gestureThreshold),(0,255,0), 5)


    if hands and buttonPressed is False:
        hand = hands[0]
        fingers = detector.fingersUp(hand)
        cx, cy = hand['center']
        # print(fingers)

        if cy<=gestureThreshold: #if hand is at the height of the face

            #Gesture 1 - Left
            if fingers == [1,0,0,0,0]:
                print('left')
                if imgNo>0:
                    imgNo -= 1
                    buttonPressed = True

            #Gesture 2 - Right
            if fingers == [0,0,0,0,1]:
                print('right')
                if imgNo < len(pathImages)-1:
                    buttonPressed = True
                    imgNo +=1
    
    #Button pressed iterations
    if buttonPressed:
        buttonCounter +=1
        if buttonCounter> buttonDelay:
            buttonPressed = False
            buttonCounter = 0

    
    #adding webcam img on slide
    imgSmall = cv2.resize(img, (ws,hs))
    h, w, _=currentImg.shape
    currentImg[0:hs,w-ws:w] = imgSmall
    cv2.imshow("Image",img)
    cv2.imshow("Slide", currentImg)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

