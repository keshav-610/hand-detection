import cv2 
import mediapipe as mp
import time

cap=cv2.VideoCapture(0)
mphands=mp.solutions.hands
hands=mphands.Hands()
mpDraw=mp.solutions.drawing_utils

ptime=0
ctime=0

while True:
    try:
        success,img=cap.read()
        imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        results=hands.process(imgRGB)
        
        if results.multi_hand_landmarks:
            for handlms in results.multi_hand_landmarks:
                for id,lm in enumerate(handlms.landmark):
                    h,w,c=img.shape
                    cx,cy=int(lm.x*w),int(lm.y*h)
                    print(id,cx,cy)
                    if id==8:
                        cv2.circle(img,(cx,cy),15,(255,0,0),cv2.FILLED)
                    if id==4:
                        cv2.circle(img, (cx, cy), 15, (255, 0, 0), cv2.FILLED)

                mpDraw.draw_landmarks(img,handlms,mphands.HAND_CONNECTIONS)


        ctime=time.time()
        fps=1/(ctime-ptime)
        ptime=ctime

        cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,2,255),3)

        cv2.imshow("Image",img)
        cv2.waitKey(1)
    
    except KeyboardInterrupt:
        print("Caught KeyboardInterrupt")
        break