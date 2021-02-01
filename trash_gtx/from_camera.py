import cv2
from main import *


darknetmain = darknet_main()

darknetmain.setGPU(is_GPU=True)

saveVid ="./data/test_sampah_tasik.mp4"
video = cv2.VideoCapture("http://ahmed:12345@100.96.1.19:8081/?action=stream")
width = int(video.get(3))
height = int(video.get(4))
size = (width,height)
result = cv2.VideoWriter('sampah_tasik.mpeg',cv2.VideoWriter_fourcc(*'MPJG'),20,size)

if video.isOpened():
    while(True):
        res, cv_img = video.read()
        if res==False:
            break
        imcaptions, boundingBoxs = darknetmain.performDetect(cv_img)
        if len(imcaptions)>0:
            for i in range(len(imcaptions)):
                cv_img = cv2.rectangle(cv_img, boundingBoxs[i][0], boundingBoxs[i][2], (0, 255, 0), 2)
                cv_img = cv2.putText(cv_img, imcaptions[i], boundingBoxs[i][0], cv2.FONT_HERSHEY_SIMPLEX, 1,
                                     (0, 0, 255))
            cv2.imshow("result", cv_img)
            result.write(cv_img)
            key = cv2.waitKey(1)
            if key==27:
                break
        else:
            print("no result")
else:
    print("Cannot read the video file")