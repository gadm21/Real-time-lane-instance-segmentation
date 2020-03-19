
import cv2
import numpy as np 

image= np.zeros((102, 100 ), dtype= np.uint8)

result= np.copy(image) 

image[10:24, 11:15]= 50
image[30:40, 90:99]= 100



_, labels, stats, _= cv2.connectedComponentsWithStats(image, connectivity= 4) 


for i, stat in enumerate(stats):
    idx= np.where(labels==i) 
    result[idx]= i* 50 + 50

cv2.imwrite("result1.png", image) 
cv2.imwrite("result2.png", result) 