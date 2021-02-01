import cv2
video = cv2.VideoCapture(0)

if video.isOpened():
    while(True):
        res, cv_img = video.read()
        if res == True:
            cv2.imshow("Test",cv_img)
        
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break

    video.release()
    cv2.destroyAllWindows()
