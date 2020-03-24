import cv2
import numpy as np 
from sklearn.cluster import DBSCAN
import os 
import sys 
from matplotlib import pyplot as plt 
sys.path.append(os.getcwd()) 
image_path= 'images'

def toGray(image):
    if len(image.shape) == 3:
        gray_image= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else: gray_image= image

    return gray_image

def remove_noise(image, min_area_threshold= 100):
    image= toGray(image)
    _, labels, stats, _= cv2.connectedComponentsWithStats(image, connectivity= 8, ltype= cv2.CV_32S)


    for index, stat in enumerate(stats):
        if stat[4] < min_area_threshold: 
            idx= np.where(labels==index)
            image[idx]= 0
    
    return image

def morphological_process(image, kernel_size= 5):
    
    assert (len(image.shape) < 3), "binary image must have a single channel"
    image= np.array(image, np.uint8) 

    kernel= cv2.getStructuringElement(shape= cv2.MORPH_ELLIPSE, ksize= (kernel_size, kernel_size))
    filled_holes= cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=1 )

    return filled_holes 

def get_hist(image):
    h= image.shape[0]
    start_h = int(0.8 * h)
    end_h = int(0.9 * h) 
    return np.sum(image[start_h:end_h, :], axis= 0) 


def truncate_tumor(image):

    def get_min_max_points():
        idx= np.where(image==255) 
        minn= np.min(idx[0])
        maxx= np.max(idx[0])
        return minn, maxx, idx 

    def get_low_indecies(minn, maxx, idx255):
        start_h= int( (maxx-minn) * 0.1) + minn 
        end_h= maxx 
        #target= np.where(idx255[idx255[0] > start_h][ :]) 
        
    

    #get the highest and lowest lane points
    minn, maxx, idx255= get_min_max_points()

    #get indecies from 90% of hieght to 100%
    low_indecies= get_low_indecies(minn, maxx, image)

    #cluster these indecies (based on their locations not their values)
    
    ys= idx255[0]
    xs= idx255[1]
    print(ys)

    ys= ys>10
    print(ys) 

    #window highet starts, also, at 90% of height and has hight 10 pixels
    #initial center of window to be determined
    #width of window should be about double the width of the lane. Thus, width of lane should be calculated

    pass


def process(image):

    image= morphological_process(image) 
    image= remove_noise(image)

    image_h= image.shape[0]
    idx= np.where(image == 255) 
    ys, xs= idx[0], idx[1]
    minn= np.min(ys) 
    window_h= int((image_h - minn) * 0.05)
    db= DBSCAN(eps= 5, min_samples= 10)


    lanes= [] 
    for h in range(minn, image_h - window_h, window_h):
        inv_h= image_h - h + minn
        target= (ys > inv_h - window_h) & (ys < inv_h) 
        target_xs, target_ys= xs[target], ys[target]
        target_pix= (target_ys, target_xs) 
        
        ret= db.fit(np.array(target_pix).transpose()) 
        
        
        
    





'''
cv2.imshow("R", image) 
cv2.waitKey(0) 
cv2.destroyAllWindows()

ret= DBSCAN(eps= 10, min_samples=10).fit(hist) 
labels= ret.labels_
labels[ labels < 0 ]= 1
'''



if __name__ == "__main__":

    image= cv2.imread(os.path.join(image_path, 'binary.png'), cv2.COLOR_BGR2GRAY)
    process(image)
    

















'''
arr= np.array([[1, 2], [2, 2], [2, 3], [8, 7], [8, 8], [25, 80]])

db= DBSCAN(eps= 3, min_samples=2)

ret= db.fit(arr) 

unique_labels= np.unique(ret.labels_)

for label in unique_labels:
    if label==-1: continue 
    sub= arr[ret.labels_==label]
    print(sub) 
'''