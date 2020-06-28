

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

weights_path = r'C:\Users\gad\Desktop\repos\VOLO\new_weights\tusimple_lanenet_vgg.ckpt'

def get_info(json_dir, json_file, start = 0, end = 10) :

    json_path = ops.join(json_dir, json_file) 
    assert ops.exists(json_path), 'json file:{} not exist'.format(json_path) 

    info = []
    with open(json_path, 'r') as file:
        for i, line in enumerate(file):

            if i < start : continue 
            if i > end : break 

            line_info = json.loads(line) 
            image_path = ops.join(json_dir, line_info['raw_file'] )
            lanes_list = line_info['lanes'] 
            h_list = line_info['h_samples']

            info.append([image_path, lanes_list, h_list])
        
    return info 

def open_image(image) :

    idx = image.nonzero()[0] 
    lowest = np.min(idx) 
    separate = np.minimum(lowest + 70 , image.shape[1]-1) 

    kernel11 = np.ones((11,11), np.uint8)
    kernel7 = np.ones((7,7), np.uint8)
    kernel5 = np.ones((5,5), np.uint8) 
    kernel3 = np.ones((3,3), np.uint8) 


    image[lowest:separate,:] = cv2.erode(image[lowest:separate,:], kernel11, iterations=1)  
    #image[separate:,:] |= eroded[separate:,:]


    image = cv2.dilate(image, kernel5, iterations= 1) 
    image = cv2.erode(image, kernel3, iterations=1)

    half = image.shape[1] //2
    image = cv2.line(image, (half,lowest), (half,separate), 255) 
    return image
    
def process_logits_and_score(logits, score):

    a = ['score1', 'score2', 'logits1', 'logits2'] 

    for i in range(logits.shape[0]) : 

        print("analyzing:{}".format(a[1]))
        analyze(score[i, :,:,1], id = i ) 
        
        #print("analyzing:{}".format(a[3]))
        #analyze(logits[i, :,:,1] )


def analyze(score2, id, gamma = 0.1):

        score2 = resize_image(score2, (1280, 720)) 

        score2[score2<0] = 0 

        score2 *= (1/np.max(score2)) 
        score2[score2 <= 0.2] = 0 
        score2[score2 > 0.2] = 255

        score2 = np.array(score2, dtype = np.uint8)
        score2 = open_image(score2) 
        
        save_image('images/crazy_idea', 'score_{}'.format(id), score2) 


def order_lanes(lanes):
    means = [] 
    for lane in lanes: means.append(np.mean(lane)) 
    perm = np.argsort(means)
    return np.array(lanes)[perm] 


def draw_lanes(image, lanes, color):
    h_samples = lanes[0] 
    lanes_x_coords = lanes[1]
    for lane in lanes_x_coords : 
        for i, x in enumerate(lane) : 
            if x == -2 : continue 

            image = cv2.circle(image, 
                              (x, h_samples[i]),
                              radius = 4,
                              color = color,
                              thickness = 4 )

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
        if len(lanes2) > i : 
            lane1 = lanes1[i] 
            lane2 = lanes2[i] 
            for point1, point2 in zip(lane1, lane2):
                diff += np.abs(point1 - point2) 
                
    return diff 


def predict( images):
    if not isinstance(images, list): images = [images] 
    original_shape = images[0].shape[0:2]

    images = np.array([normalize(resize_image( image , (512, 256))) for image in images])

    x = tf.placeholder(name = 'input', dtype = tf.float32, shape = [images.shape[0], 256, 512, 3])
    net = LaneNet('test') 
    binary, logits, score = net.inference(x, 'lanenet_model')

    with tf.Session() as sess : 
        load_weights(sess, weights_path) 
        binary_output, logits_output, score_output = sess.run([binary, logits, score], {x : images})


    process_logits_and_score(logits_output, score_output) 

    binary_images = [] 
    for i in range(len(images)) :
        binary_image = np.array(binary_output[i, :, :] * 255, dtype = np.uint8)
        binary_image = resize_image(binary_image, (original_shape[1], original_shape[0]))  
        binary_images.append(binary_image )  

    return binary_images  # [0:255] not [0:1]


if __name__ == "__main__" :
    
    
    infos = get_info(json_dir, json_file, 361,366)
    #binary_images = [normalize(to_gray(read_image(i))) for i in get_image_paths_list(images_path2)] 

    postprocessor = PostProcessor() 
    #his_postprocessor = LaneNetPostProcessor() 

    my_time_cost = []
    #his_time_cost = [] 

    images = [read_image(infos[i][0]) for i in range(len(infos))] 
    
    binary_images = predict(images) 
    
    
    for i, b in enumerate(binary_images) : 
        save_image('images/crazy_idea', 'binary_{}'.format(i), b)
        #show_image(b) 
    '''
    for i, info in enumerate(infos):
        image_path, lane_list, h_samples = info
        image = read_image(image_path)
        binary_image = binary_images[i]

        #time1 = time.time() 
        #his_ret = his_postprocessor.postprocess(binary_image, instance_image, image) 
        time2 = time.time()
        my_ret = postprocessor.process(binary_image, image)
        time3 = time.time() 

        gt_lanes = order_lanes(lane_list) 
        my_lanes = order_lanes(get_lanes(h_samples, my_ret['lanes_params'])) 
        #his_lanes = order_lanes(get_lanes(h_samples, his_ret['lanes_params'])) 

        #my_score = compare_lanes(gt_lanes, my_lanes) 
        #his_score = compare_lanes(gt_lanes, his_lanes) 
        
        my_time_cost.append(time3-time2) 
        #his_time_cost.append(time2-time1) 
        
        draw_lanes(image, (h_samples, my_lanes), color = [255, 0, 0])
        draw_lanes(image, (h_samples, gt_lanes), color = [0, 0, 255])

        save_image('images/new_result', 'lanes_on_source_{}'.format(i), image)
        #save_image('images/result', 'binary_{}'.format(i), binary_image*255)
        #save_image('images/result', 'mine_{}'.format(i), my_ret['mask_image']) 
        #save_image('images/result', 'his_{}'.format(i), his_ret['mask_image'])


        if i % 3 == 0 :
            print("my postprocessing time :{}".format(np.mean(my_time_cost))) 
            #print("his postprocessing time:{}".format(np.mean(his_time_cost)))
    '''
        
            