

import cv2 
import numpy as np 
from sklearn.cluster import DBSCAN
import os 
import sys 
import random
random.seed(4)
sys.path.append(os.getcwd()) 

def read_image(image_path):
    return cv2.imread(image_path) 

def save_image(save_dir, image_name, image):
    os.makedirs(save_dir, exist_ok=True) 
    cv2.imwrite(os.path.join(save_dir, image_name+".png"), image)

def show_image(image, label= 'r' ):

    cv2.imshow(label, image) 
    cv2.waitKey(0)  
    cv2.destroyWindow(label)

def to_gray(image):

    if len(image.shape) == 3:
        gray_image= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else: gray_image= image 

    return gray_image 

def to_colored(image):
    if len(image.shape) < 3:
        colored_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else: colored_image= image

    return colored_image 

def remove_noise(image, min_area_threshold= 500):
    assert len(image.shape) != 3, "remove noise accepts gray images only"
    _, labels, stats, _= cv2.connectedComponentsWithStats(image, connectivity= 8, ltype= cv2.CV_32S)

    for index, stat in enumerate(stats):
        if stat[4] < min_area_threshold:
            noise_indecies= np.where(labels == index)
            image[noise_indecies] = 0

    return image 

def morphological_process(image, kernel_size= 5):
    assert len(image.shape) != 3, "morphological_process accepts gray images only"

    kernel= cv2.getStructuringElement(shape= cv2.MORPH_ELLIPSE, ksize= (kernel_size, kernel_size)) 
    filled_holes_image= cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations= 1)

    return filled_holes_image 
















class Lane(object):

    def __init__(self):
        self.clusters = []
        self.window_h= 5 
    
    def mean(self):
        current_mean= int(np.mean(self.clusters[-1][1]))

        if len(self.clusters) > 1 :
            prev_mean= int(np.mean(self.clusters[-2][1]))
        else: prev_mean= current_mean 

        mean_change= current_mean - prev_mean 

        return current_mean + mean_change 
        
        

    def num_points(self):
        total= 0
        for cluster in self.clusters:
            total+= cluster[0].shape[0]
        return total 
    
    def colorize(self, image, color):
        for cluster in self.clusters:
            image[cluster]= color 

        return image 

    def cluster_width(self):
        total= 0
        for cluster in self.clusters:
            low_x = np.min(cluster[1])
            high_x= np.max(cluster[1])       
            diff= high_x - low_x
            total+= diff 
        
        return int(total // len(self.clusters) )

    def blacken(self, image) :
        for cluster in self.clusters:
            image[cluster] = 0 
        
        return image 

    def complete(self, cluster_coords, image):

        self.clusters.append(cluster_coords) 

        image_h, image_w = image.shape
        lanes_coords= np.where(image == 255) 
        lowest_lane_coord= np.min(lanes_coords[0])
        highest_lane_coord= np.max(cluster_coords[0]) 
        window_center= self.mean()
        
        
        for window in range(highest_lane_coord, lowest_lane_coord, - self.window_h):
            m= self.mean()
            if m : window_center= m
            window_w= self.cluster_width()

            window_pix= (lanes_coords[0] >= window - self.window_h) & \
                        (lanes_coords[0] < window) & \
                        (lanes_coords[1] > (window_center - window_w)) & \
                        (lanes_coords[1] < (window_center + window_w))
            lane_coords_within_window = (lanes_coords[0][window_pix], lanes_coords[1][window_pix])
            if lane_coords_within_window[0].shape[0] == 0 : continue  

            self.clusters.append(lane_coords_within_window) 

        image= self.blacken(image) 
        image= remove_noise(image) 
        return image 
             




class PostProcessor(object):

    def __init__(self):
        
        self.stride_h= -5
        
        self.color_map= [[0, 0, 255],
                        [0, 255, 0],
                        [255, 0, 0],
                        [255, 255, 0]]


        self.dbscan_eps= 8
        self.dbscan_min_samples= 30
        self.db= DBSCAN(self.dbscan_eps, self.dbscan_min_samples) 
    
    def pre_processing(self, image):
        image= to_gray(image) 
        image= remove_noise(image) 
        image= morphological_process(image) 
        return image 


    def apply_clustering_on_stride(self, coords):

        ret= self.db.fit(np.array(coords).transpose())
        labels= ret.labels_
        unique_labels= np.unique(labels) 
        return labels, unique_labels 

    def post_process(self, image):
        
        image= self.pre_processing(image) 

        image_h, image_w = image.shape
        lanes_coords= np.where(image == 255) 
        lowest_lane_coord= np.min(lanes_coords[0])
        highest_lane_coord= np.max(lanes_coords[0]) 
        
        
        lanes= []
        for stride in range(highest_lane_coord, lowest_lane_coord, self.stride_h):
            lanes_coords= np.where(image == 255)
            target_within_stride= (lanes_coords[0] < stride) & (lanes_coords[0] >= (stride + self.stride_h))
            stride_lanes_coords= (lanes_coords[0][target_within_stride], lanes_coords[1][target_within_stride])
            
            if stride_lanes_coords[0].shape[0] == 0 : continue 

            labels, unique_labels= self.apply_clustering_on_stride(stride_lanes_coords) 
            for label in unique_labels:
                if label==-1:  continue             
                cluster= (labels == label)
                cluster_coords= (stride_lanes_coords[0][cluster], stride_lanes_coords[1][cluster])
                
                lane= Lane() 
                image= lane.complete(cluster_coords, image) 
                lanes.append(lane) 
        
        print("done")

        image= to_colored(image) 
        for lane in lanes:
            image= lane.colorize(image, self.color_map[random.randint(0, 3)])
                
        return image 








