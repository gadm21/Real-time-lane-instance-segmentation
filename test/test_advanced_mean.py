



import os 
import sys 
sys.path.append(os.getcwd())

import numpy as np 
import cv2 
import pickle 
import tensorflow as tf 
import time     
import glob 

from LaneNet_model.LaneNet import LaneNet 
from LaneNet_model.LaneNet_PostProcessor import LaneNetPostProcessor
from LaneNet_model.my_postprocessor import * 

from test_utils import * 

images_path = 'images/input'
images_path2 = 'images/input_binary'
weights_path = r'C:\Users\gad\Desktop\repos\VOLO\weights\tusimple_lanenet_vgg.ckpt'



def to_gray(image):

    if len(image.shape) == 3:
        gray_image= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else: gray_image= image 

    return gray_image 

def remove_noise(image, min_area_threshold= 500):
    assert len(image.shape) != 3, "remove noise accepts gray images only"
    
    _, labels, stats, _= cv2.connectedComponentsWithStats(image, connectivity= 8, ltype= cv2.CV_32S)

    for index, stat in enumerate(stats):
        if stat[4] < min_area_threshold:
            noise_indecies= np.where(labels == index)
            image[noise_indecies] = 0

    return image 

def cluster_strides( coords):

    dbscan_eps= 8
    dbscan_min_samples= 30
    db= DBSCAN(dbscan_eps, dbscan_min_samples) 

    ret= db.fit(np.array(coords).transpose())
    labels= ret.labels_
    unique_labels= np.unique(labels) 
    return labels, unique_labels 




class Lane(object):

    def __init__(self, id=0):
        self.color_map= [[0, 0, 255],
                        [0, 255, 0],
                        [255, 0, 0],
                        [255, 255, 0]]
        self.clusters = []
        self.means= [] 
        self.window_h= 5 
        self.id= id 
        self.valid= False 
        self.image_h= None 
        self.image_w= None 
        self.remap_to_x= None 
        self.remap_to_y= None 
        self.lane_curve = None 
        self.birdeye_params = None 
    
    def mean(self):
        current_mean= (int(np.mean(self.clusters[-1][0])), int(np.mean(self.clusters[-1][1])))
        prev_mean= self.means[-1]

        mean_change= (current_mean[0] - prev_mean[0], current_mean[1] - prev_mean[1]) 
        predicted_mean= (current_mean[0] + mean_change[0] , current_mean[1] + mean_change[1])
        self.means.append(predicted_mean) 

        return predicted_mean    

    def advanced_mean(self):
        current_mean= (np.mean(self.clusters[-1][0]), np.mean(self.clusters[-1][1]))
        prev_mean= self.means[-1]
        mean_change= (current_mean[0] - prev_mean[0], current_mean[1] - prev_mean[1]) 

        new_mean = (current_mean[0] + mean_change[0], int(0.2*prev_mean[1] + 0.8*current_mean[1]))
        self.means.append(new_mean) 
        return new_mean 

    def print_info(self):
        print("lane {:d} info:".format(self.id))
        print("number of pixels on this lane == {:d}".format(self.num_points()))
        print("average cluster width == {:f}".format(self.cluster_width()))
        
        print("mumber of clusters == {:d}".format(len(self.clusters)))
        print("____________________________________________")
        print() 

    def num_points(self):
        total= 0
        for cluster in self.clusters:
            total+= cluster[0].shape[0]
        return int(total)  
    
    def draw_mask(self, shape= None, color_means= False):
        if shape is None: shape= ( self.image_h, self.image_w,3 ) 

        mask= np.zeros(shape= shape, dtype= np.uint8)
        color = self.color_map[self.id%len(self.color_map)]

        if color_means:
            for cluster in self.clusters:

                mean_x = np.mean(cluster[1], dtype=np.int32) 
                mean_y = np.mean(cluster[0], dtype=np.int32) 
                
                cv2.circle(mask, (mean_x, mean_y), 1, color, 2)

        else:
            for cluster in self.clusters:
                mask[cluster]= color 

        return mask 

    def cluster_width(self):
        total= 0
        for cluster in self.clusters:
            low_x = np.min(cluster[1])
            high_x= np.max(cluster[1])       
            diff= high_x - low_x
            total+= diff 
        
        last_width = int(np.max(self.clusters[-1][1]) - np.min(self.clusters[-1][1]))
        average_width = int(total // len(self.clusters) )
        weighted_width = int(last_width) #int(0.2 *average_width + 0.8 * last_width)
        return weighted_width 

    def blacken(self, image) :
        for cluster in self.clusters:
            image[cluster] = 0
    
        return image 

    def complete(self, cluster_coords, image):

        self.clusters.append(cluster_coords) 
        self.means.append((np.mean(self.clusters[-1][0]), np.mean(self.clusters[-1][1])))

        self.image_h, self.image_w = image.shape[0], image.shape[1]
        lanes_coords= np.where(image == 255) 
        
        lowest_lane_coord= np.min(lanes_coords[0])
        highest_lane_coord= np.max(lanes_coords[0]) 
        window_center= self.means[-1][1]
        
        
        for window in range(highest_lane_coord, lowest_lane_coord, -5):
            margin= int(self.cluster_width())

            window_pix= (lanes_coords[0] >= window - 5) & \
                        (lanes_coords[0] < window) & \
                        (lanes_coords[1] > (window_center - margin)) & \
                        (lanes_coords[1] < (window_center + margin))
            lane_coords_within_window = (lanes_coords[0][window_pix], lanes_coords[1][window_pix])
            if lane_coords_within_window[0].shape[0] == 0 : continue  

            self.clusters.append(lane_coords_within_window) 
            window_center= self.mean()[1]

        image= self.blacken(image) 
        return image 
     
  
def inspect_lanes(lanes):
    total_points= 0
    for lane in lanes:
        total_points+= lane.num_points()
    average_lane_points= total_points / len(lanes)
    min_lane_points= average_lane_points * 0.4

    for lane in lanes:
        if lane.num_points() > min_lane_points:
            lane.valid= True 





def process( image, source):
    id = 0 

    image = np.array(image*255, dtype = np.uint8)
    image = resize_image(image, source.shape[0:2])
    image= remove_noise(image) 

    lanes_coords= np.where(image == 255) 
    assert len(lanes_coords[0]), 'no lanes to process' 

    lowest_lane_coord= np.min(lanes_coords[0])
    highest_lane_coord= np.max(lanes_coords[0]) 
    
    lanes= []
    for stride in range(highest_lane_coord, lowest_lane_coord, -5):
        lanes_coords= np.where(image == 255)
        target_within_stride= (lanes_coords[0] < stride) & (lanes_coords[0] >= (stride + -5))
        stride_lanes_coords= (lanes_coords[0][target_within_stride], lanes_coords[1][target_within_stride])
        if len(stride_lanes_coords[0]) == 0 : continue 
        
        labels, unique_labels= cluster_strides(stride_lanes_coords) 
        for label in unique_labels:
            if label==-1:  continue             
            cluster= (labels == label)
            cluster_coords= (stride_lanes_coords[0][cluster], stride_lanes_coords[1][cluster])
            
            lane= Lane(id) 
            id += 1
            image= lane.complete(cluster_coords, image) 
            lanes.append(lane) 
            
    inspect_lanes(lanes) 

    mask = np.zeros((image.shape[0], image.shape[1], 3), dtype = np.uint8) 
    for lane in lanes:
        if not lane.valid : continue 
        mask |= lane.draw_mask(shape = mask.shape, color_means = False) 
    return mask 


def predict( images):
    if not isinstance(images, list): images = [images] 
    original_shape = images[0].shape[0:2]

    images = np.array([normalize(resize_image( image , (512, 256))) for image in images])

    x = tf.placeholder(name = 'input', dtype = tf.float32, shape = [images.shape[0], 256, 512, 3])
    net = LaneNet('test') 
    binary, instance= net.inference(x, 'lanenet_model')

    with tf.Session() as sess : 
        load_weights(sess, weights_path) 
        binary_output, instance_segmentation_output = sess.run([binary, instance], {x : images})
    
    

    binary_images = [] 
    instance_images = [] 

    for i in range(len(images)) :
        binary_image = binary_output[i, :, :] 
        instance_image = instance_segmentation_output[i, :, :, :]  
        binary_images.append(binary_image) 
        instance_images.append(instance_image) 

    return binary_images, instance_images 
    #return binary_output, instance_segmentation_output  # [0,1] not [0,255]








def test(binary_images, instance_images, source_images):
    #if not isinstance(binary_images, list) : binary_images = [binary_images] 
    #if not isinstance(instance_images, list): instance_images = [instance_images] 
    #if not isinstance(source_images, list): source_images = [source_images]
    
    
    postprocessor = PostProcessor() 
    his_postprocessor = LaneNetPostProcessor() 
    for i in range(len(source_images)):

        his_ret = his_postprocessor.postprocess(binary_images[i], instance_images[i], source_images[i]) 
        ret = postprocessor.process(binary_images[i], source_images[i])    
        local_mask = process(binary_images[i], source_images[i]) 

        save_image('images/result', 'my_lane_mask_{}'.format( i), ret['mask_image']) 
        save_image('images/result', 'his_lane_mask_{}'.format(i), his_ret['mask_image']) 
        save_image('images/result', 'local_mask_{}'.format(i), local_mask) 


















if __name__ == "__main__":

    #binary_images = [to_gray(read_image(i)) for i in get_image_paths_list(images_path2)]

    source_images = [resize_image(read_image(i), (720, 1280)) for i in get_image_paths_list(images_path)][0:1]
    

    binary_images, instance_images = predict(source_images) 
    
    test(binary_images, instance_images, source_images)  