
import cv2
import numpy as np 
from sklearn.cluster import DBSCAN

arr= np.array([[1, 2], [2, 2], [2, 3], [8, 7], [8, 8], [25, 80]])

db= DBSCAN(eps= 3, min_samples=2)

ret= db.fit(arr) 

print(ret.components_) 

