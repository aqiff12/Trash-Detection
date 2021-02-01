import cv2
from main import *

videoPath = "./data/test_sampah1.mp4"

darknetmain = darknet_main()

darknetmain.setGPU(is_GPU=True)

video = cv2.VideoCapture(videoPath)
width = int(video.get(3))
height = int(video.get(4))
size = (width,height)
result = cv2.VideoWriter('sampah1_output.mpeg',cv2.VideoWriter_fourcc(*'MPJG'),20,size)

if video.isOpened():
    while(True):
        res, cv_img = video.read()
        if res==False:
            break
        imcaptions, boundingBoxs = darknetmain.performDetect(cv_img)
        if len(imcaptions)>0:
            if len(imcaptions) > 0:
                for i in range(len(imcaptions)):
                    name = imcaptions[i]
                    name = name[:5]
                    print(name + " is found")
                    cv_img = cv2.rectangle(cv_img, boundingBoxs[i][0], boundingBoxs[i][2], (0, 255, 0), 2)
                    cv_img = cv2.putText(cv_img, imcaptions[i], boundingBoxs[i][0], cv2.FONT_HERSHEY_SIMPLEX, 1,
                                         (0, 0, 255))         
            cv2.imshow("result", cv_img)
            result.write(cv_img)
            cv2.waitKey(1)
        else:
            print("no result")
    result.release()
    video.release()
else:
    print("Cannot read the video file")