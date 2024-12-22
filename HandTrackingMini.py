import cv2
import mediapipe as mp
import time

#create video object
cap=cv2.VideoCapture(0)

mpHands = mp.solutions.hands
# hands only uses RGB image
hands = mpHands.Hands()
# mediapipe method to draw all these point
mpDraw = mp.solutions.drawing_utils

#1 frame rate
pTime = 0
cTime = 0

while True:
    # run a webcame ---- giveing a frame
    success,img=cap.read()
    img=cv2.flip(img,1)
    # sending image
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)

    # extracking multiple hands if we have, we extract one by one
    if results.multi_hand_landmarks:
        # hand land marks " handLms "
        for handLms in results.multi_hand_landmarks:

            for id,lm in enumerate(handLms.landmark):

                # give the index or id and landmarks of the points it will give ratio
                # print(id,lm)
                
                # hight , width , channel
                h, w, c=img.shape
                # find position... it will give cx and cy position
                cx,cy=int(lm.x*w),int(lm.y*h)
                print(id,cx,cy)
                # if id==4:
                cv2.circle(img,(cx,cy),8,(255,0,255),cv2.FILLED)

            # HAND_CONNECTION will draw a connection between points
            mpDraw.draw_landmarks(img,handLms,mpHands.HAND_CONNECTIONS)

    # 2 frame rate
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    # cv2.putText(our image,convert into str and int for no decimal,position value,font,scale,color,thickness)
    cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)

    
    cv2.imshow("Image",img)
    cv2.waitKey(1)