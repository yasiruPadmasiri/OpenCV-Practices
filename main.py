import cv2

cap= cv2.VideoCapture("highway.mp4")

object_detector=cv2.createBackgroundSubtractorMOG2()
while True:

    ret,frame=cap.read()

    #    object detector

    mask=object_detector.apply(frame)


    Contours,_=cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    for cnt in  Contours:
        cv2.drawContours(frame,[cnt],-1,(0,225,0),2)


    cv2.imshow('frane',frame)
    cv2.imshow("mask",mask)
    key=cv2.waitKey(30)
    if key==27:
        break
cap.release()
cv2.destroyAllWindows()





cv2.waitKey(0)
