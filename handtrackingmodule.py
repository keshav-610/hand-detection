import cv2
import mediapipe as mp
import time

class handdetector:
    def __init__(self,mode=False,maxhands=2,complexity=1,detectioncon=0.5,trackcon=0.5):
        self.mode=mode
        self.maxhands=maxhands
        self.complexity=complexity
        self.detectioncon=detectioncon
        self.trackcon=trackcon

        self.mphands = mp.solutions.hands
        self.hands = self.mphands.Hands(self.mode,self.maxhands,self.complexity,self.detectioncon,self.trackcon)
        self.mpDraw = mp.solutions.drawing_utils


    def findhands(self,img,draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handlms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handlms, self.mphands.HAND_CONNECTIONS)

        return img

    def findposition(self,img,handno=0,draw=True):
        lmlist=[]
        if self.results.multi_hand_landmarks:
            myhand=self.results.multi_hand_landmarks[handno]
            for id, lm in enumerate(myhand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                #print(id, cx, cy)
                lmlist.append([id,cx,cy])
                if draw:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 0), cv2.FILLED)

            return lmlist







def main():
    ptime = 0
    ctime = 0
    cap = cv2.VideoCapture(0)
    detector=handdetector()

    while True:
        success, img = cap.read()
        img=detector.findhands(img)
        lmlist= detector.findposition(img)
        if len(lmlist)!=0:
            print(lmlist[4])
        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 2, 255), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(1)



if __name__=="__main__":
    main()