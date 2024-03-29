import cv2 
import mediapipe as mp      #import libraries
import time

cap=cv2.VideoCapture(0)  #Recognize the webcam, default is 0
mphands=mp.solutions.hands #Create a object for import hand module from solutions package for access the functionalities provided by hand module include hand tracking
hands=mphands.Hands()   #Initializes the hand tracking model used to detect & track hands in image or video
mpDraw=mp.solutions.drawing_utils #provides utilities for drawing landmarks, connections and other functions n images or video

ptime=0
ctime=0

while True:
    try:
        success,img=cap.read()  #success represents a boolean value whether the frame was read; img represents actual image was read but we use while true for capturing multiple frames, considers as video
        imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB) # converts the color of video to RGB
        results=hands.process(imgRGB) # analyzes the provided image frame img RGB & Detects hands
        
        if results.multi_hand_landmarks: #check if the list is not empty
            for handlms in results.multi_hand_landmarks:    #iterates over each set of hand landmarks detected in frame
                for id,lm in enumerate(handlms.landmark): #id represents the index of the landmarks detected in frame; lm represents the position of the landmark, iterates over each individual lm within the detected hands
                    h,w,c=img.shape #retrieves the height , width and number of channels of image img.
                    cx,cy=int(lm.x*w),int(lm.y*h) #calculates the pixel coordinates of landmark relative to width and height. It multiples the normalizes coordinates between 0 & 1
                    print(id,cx,cy) 
                    if id==8: #the 8th id is th tip of index finger
                        cv2.circle(img,(cx,cy),15,(255,0,0),cv2.FILLED) #the landmark is a blue circle
                    if id==4:
                        cv2.circle(img, (cx, cy), 15, (255, 0, 0), cv2.FILLED)

                mpDraw.draw_landmarks(img,handlms,mphands.HAND_CONNECTIONS) 
                """"Three arguments 
                a)image
                b)landmarks of set landmarks
                c)Specifies how to connect the landmarks to visualize the hand structure"""

        ctime=time.time() #retrieves the current time in seconds using time module
        fps=1/(ctime-ptime) 
        ptime=ctime

        cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,2,255),3) #draw text on image to display FPS

        cv2.imshow("Image",img)
        cv2.waitKey(1)
    
    except KeyboardInterrupt:
        print("Caught KeyboardInterrupt")
        break
    
    