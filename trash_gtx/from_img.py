import cv2
from main import *

imagePath =  "./data/trash2.png"
newName = imagePath[7:-4] + "_new.png"

darknetmain = darknet_main()
darknetmain.setGPU(is_GPU=True)

cv_img = cv2.imread(imagePath)
imcaptions, boundingBoxs = darknetmain.performDetect(cv_img)
if len(imcaptions)>0:
    for i in range(len(imcaptions)):
        cv_img = cv2.rectangle(cv_img, boundingBoxs[i][0], boundingBoxs[i][2], (0, 255, 0), 2)
        cv_img = cv2.putText(cv_img, imcaptions[i], boundingBoxs[i][0], cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
    cv2.imshow("result", cv_img)
    cv2.imwrite("./data/"+newName,cv_img)
    cv2.waitKey(0)
else:
    print("no result")