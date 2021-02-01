import cv2
k = cv2.imread("bottle_8.png")
print(k.shape[:2])
image = k[0:397, 50:822]
cv2.imwrite("Cropped.png",image)