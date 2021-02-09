from IoU import intersection_over_union
import cv2
import numpy as numpy
import torch

img = cv2.imread(r"Image\img_1.jpg")

imgResize = cv2.resize(img, (600,500))
BB_1 = torch.tensor([[180 , 20, 450 , 500]])
BB_2 = torch.tensor([[20 , 30, 200 , 240]])

IoU = intersection_over_union(BB_2, BB_1, box_formats="corners")

cv2.rectangle(imgResize,(BB_1[0,0],BB_1[0,1]),(BB_1[0,2], BB_1[0,3]),(0,0,255),2)
cv2.rectangle(imgResize,(BB_2[0,0],BB_2[0,1]),(BB_2[0,2], BB_2[0,3]),(255,0,0),2)

print(f"The IoU is: {IoU}")
# def on_click(event, x, y, p1, p2):
#     if event == cv2.EVENT_LBUTTONDOWN:
#         print(x,",",y)

cv2.imshow("image", imgResize)
# cv2.setMouseCallback('image', on_click)
cv2.waitKey(0)

