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

class Lane(object):

    def __init__(self, cluster_pts):
        self.clusters= [cluster_pts]
        self.mean= self.calculate_mean()
        self.lane_xs= []
        self.lane_ys= []

    def check(self, image):
        new_image= np.copy(image) 
        cv2.circle(new_image, (self.mean[1], self.mean[0]), 5, 255, 3)
        cv2.circle(new_image, (self.mean[1], self.mean[0]), 10, 255, 3)
        
        cv2.imshow('check', new_image) 
        cv2.waitKey(0) 
        cv2.destroyAllWindows()

    def calculate_mean(self):
        return (int(np.mean(self.clusters[-1][0])), int(np.mean(self.clusters[-1][1])))

    def clean(self, image, minn, window_h, start_h, xs, ys, window_w= 10, margin= 2):
        
        #image= cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) 
        mean= self.mean[1] 
        for h in range(start_h, minn, -window_h):
            target= (ys >= h - window_h) & (ys < h) #& (xs > (mean - window_w)) & (xs < (mean + window_w))
            target_xs, target_ys= xs[target], ys[target]
            
            cluster_mean= np.mean(target_xs)
            mean= cluster_mean

            image[(target_ys, target_xs)]= 0
            continue 

            
            print("left:", cluster_mean) 
            left_margin= int(cluster_mean + margin)
            right_margin= int(cluster_mean - margin) 
            
            out= (target_xs < right_margin) & (target_xs > left_margin) 
            print(np.count_nonzero(out), end=" ")
            out_ys, out_xs= target_ys[out], target_xs[out]
            
            image[(out_ys, out_xs)]=0
            #image[(saved_ys, saved_xs)]=255
        return image
        
        
    def add(self, cluster):
        self.clusters.append(cluster) 

    def color(self, image) :
        for cluster in self.clusters:
            image[cluster]= [255, 255, 0]
        

        
def distance(lane, cluster_pts):
    c_mean= (int(np.mean(cluster_pts[0])), int(np.mean(cluster_pts[1])))
    l_mean= lane.mean 

    y_abs= np.abs(c_mean[0]-l_mean[0])
    x_abs= np.abs(c_mean[1]-l_mean[1])

    return np.sqrt(y_abs**2 + x_abs**2) 

def process(image):

    image= morphological_process(image) 
    image= remove_noise(image)

    image_h= image.shape[0]
    idx= np.where(image == 255) 
    ys, xs= idx[0], idx[1]
    minn= np.min(ys) 
    window_h= int((image_h - minn) *    0.01)
    db= DBSCAN(eps= 10, min_samples= 10)


    lanes= [] 
    distance_threshold= 4 * window_h
    for h in range(image_h, minn, -window_h):
        idx= np.where(image == 255) 
        ys, xs= idx[0], idx[1]
        target= (ys >= h - window_h) & (ys < h) 
        target_xs, target_ys= xs[target], ys[target]
        if len(target_xs) == 0: continue 
        target_pix= (target_ys, target_xs) 
        
        ret= db.fit(np.array(target_pix).transpose()) 
        labels= ret.labels_
        unique_labels= np.unique(labels) 
        print(image[target_pix])
        print() 
        print(labels) 
        print() 
        print(unique_labels) 
        print()

        for label in unique_labels:
            if label == -1: continue 
            cluster_idx= np.where(labels==label)
            cluster_xs, cluster_ys= target_xs[cluster_idx], target_ys[cluster_idx] 
            cluster_pts= (cluster_ys, cluster_xs) 
            
            new_lane= Lane(cluster_pts) 
            new_lane.check(image) 
            image= new_lane.clean(image, minn, window_h, h, xs, ys) 
            lanes.append(new_lane)

        image= remove_noise(image) 
        cv2.imshow("after each height", image)
        cv2.waitKey(0) 
        cv2.destroyAllWindows()
            
    
    
    image= cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) 
    for lane in lanes:
        lane.color(image) 
    '''
    cv2.imshow("R", image) 
    cv2.waitKey(0) 
    cv2.destroyAllWindows()
    '''


    
































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