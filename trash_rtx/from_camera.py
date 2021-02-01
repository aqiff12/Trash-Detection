import cv2
from main import *
import imutils
import math

darknetmain = darknet_main()

darknetmain.setGPU(is_GPU=True)

saveVid ="./data/test_sampah_tasik.mp4"
video = cv2.VideoCapture("http://ahmed:12345@100.96.1.19:8081/?action=stream")
#video = cv2.VideoCapture(0)
width = 480
height = 180
size = (width,height)
result = cv2.VideoWriter('21_output2.avi',cv2.VideoWriter_fourcc(*'XVID'),20,size)
# initialize the known distance from the camera to the object, which
# in this case is 24 inches
KNOWN_DISTANCE = 12.0
# initialize the known object width, which in this case, the piece of
# paper is 11.81 inches wide
KNOWN_WIDTH = 12.0


if video.isOpened():
    while(True):
        res, cv_img = video.read()
        if res==True:
            k = cv_img[180:360, 90:570]
            imcaptions, boundingBoxs = darknetmain.performDetect(k)
            if len(imcaptions)>0:
                for i in range(len(imcaptions)):
                    k = cv2.rectangle(k, boundingBoxs[i][0], boundingBoxs[i][2], (0, 255, 0), 2)
                    k = cv2.putText(k, imcaptions[i], boundingBoxs[i][0], cv2.FONT_HERSHEY_SIMPLEX, 1,
                                        (0, 0, 255))
                    
                    startPoint = (180,570)
                    dividePoint = [x for x in boundingBoxs[i][2]]
                    dividePoint[0] = dividePoint[0] - 50
                    dividePoint[1] = dividePoint[1] - 50
                    if dividePoint[0] > 160:
                        print("Right")
                    elif dividePoint[0] <= 160:
                        print("Left")
                    centerPoint= (int(dividePoint[0]),int(dividePoint[1]))

                    KNOWN_DISTANCE = math.sqrt(((startPoint[0]-dividePoint[0])**2)+((startPoint[1]-dividePoint[1])**2))
                    KNOWN_WIDTH = boundingBoxs[i][2][1] - boundingBoxs[i][0][1]
                    KNOWN_HEIGHT = boundingBoxs[i][3][0] - boundingBoxs[i][0][0]
                    distance = (2 * 3.14 * 180) / (KNOWN_WIDTH + KNOWN_HEIGHT * 360) * 1000 + 3 ### Distance measuring in Inch
                    feedback = (" At {} ".format(round(distance))+"Inches")
                    print(feedback)
                    
                    k = cv2.line(k, startPoint, centerPoint,  (0, 255, 0), 2)
                result.write(k)
                cv2.imshow("result", k)
            else:
                #print("no result")
                result.write(k)
                cv2.imshow("result", k)
            
            key = cv2.waitKey(1)
            if key==27 or 0xFF == ord('q'):
                break
            
        else:
            break
    result.release()
    video.release()
    cv2.destroyAllWindows()
else:
    print("Cannot read the video file")

