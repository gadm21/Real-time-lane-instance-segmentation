

import os 
import os.path as ops 
import sys 
sys.path.append(os.getcwd())

import json 
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
my_result_path = 'images/my_result'
his_result_path = 'images/his_result' 

json_dir = r'C:\Users\gad\Desktop\data\train\clean_start'
json_file = 'label_data_0313.json'

weights_path = r'C:\Users\gad\Desktop\repos\VOLO\weights\tusimple_lanenet_vgg.ckpt'

def get_info(json_dir, json_file, num_images = 10) :

    json_path = ops.join(json_dir, json_file) 
    assert ops.exists(json_path), 'json file:{} not exist'.format(json_path) 

    info = []
    with open(json_path, 'r') as file:
        for i, line in enumerate(file):
            if i == num_images : break 

            line_info = json.loads(line) 
            image_path = ops.join(json_dir, line_info['raw_file'] )
            lanes_list = line_info['lanes'] 
            h_list = line_info['h_samples']

            info.append([image_path, lanes_list, h_list])
        
    return info 

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


def order_lanes(lanes):
    means = [] 
    for lane in lanes: means.append(np.mean(lane)) 
    perm = np.argsort(means)
    return np.array(lanes)[perm] 

def get_lanes(h_samples, lanes_params) :
    lanes = []
    coords_y = np.int_(h_samples) 
    for lane_params in lanes_params:
        coords_x = np.int_(np.clip(lane_params[0]*coords_y**2 + lane_params[1]*coords_y + lane_params[2], 0, 1280-1))
        lanes.append(coords_x)
    
    return lanes 

def compare_lanes(lanes1, lanes2 ):
    diff = 0 
    for i in range(len(lanes1)):
        diff += np.abs(lanes1[i]- lanes2[i]) 
    return diff 

if __name__ == "__main__" :
    
    infos = get_info(json_dir, json_file, 1)
    postprocessor = PostProcessor() 
    his_postprocessor = LaneNetPostProcessor() 
    my_time_cost = []
    his_time_cost = [] 

    images = [read_image(infos[i][0]) for i in range(len(infos))] 
    
    binary_images, instance_images = predict(images) 
    
    print("DONE PREDICTION !")


    for i, info in enumerate(infos):
        image_path, lane_list, h_samples = info
        image = read_image(image_path)
        binary_image = binary_images[i]
        instance_image = instance_images[i]  

        time1 = time.time() 
        my_ret = postprocessor.process(binary_image, image)
        time2 = time.time()
        his_ret = his_postprocessor.postprocess(binary_image, instance_image, image) 
        time3 = time.time() 

        gt_lanes = order_lanes(lane_list) 
        my_lanes = order_lanes(get_lanes(h_samples, my_ret['lanes_params'])) 
        his_lanes = order_lanes(get_lanes(h_samples, his_ret['lanes_params'])) 

        my_score = compare_lanes(gt_lanes, my_lanes) 
        his_score = compare_lanes(gt_lanes, his_lanes) 
        
        my_time_cost.append(time2-time1) 
        his_time_cost.append(time3-time2) 

        save_image('images/result', 'source_{}'.format(i), image)
        save_image('images/result', 'binary_{}'.format(i), binary_image*255)
        save_image('images/result', 'mine_{}'.format(i), my_ret['mask_image']) 
        save_image('images/result', 'his_{}'.format(i), his_ret['mask_image'])


        if i % 5 == 0 :
            print("my postprocessing time :{}".format(np.mean(my_time_cost))) 
            print("his postprocessing time:{}".format(np.mean(his_time_cost)))
        
        

            