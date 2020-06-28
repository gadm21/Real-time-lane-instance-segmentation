
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



images_path2 = 'images/input_binary'
my_result_path = 'images/my_result'
his_result_path = 'images/his_result' 

json_dir = r'C:\Users\gad\Desktop\data\train\clean_start'
json_file = 'label_data_0313.json'

weights_path = r'C:\Users\gad\Desktop\repos\VOLO\new_weights\tusimple_lanenet_vgg.ckpt'

predictions_path = 'images2/scores'


def load_weights(sess, weights_path):
    saver = tf.train.Saver() 
    saver.restore(sess = sess, save_path= weights_path) 



def get_image_paths_list(images_path):
    images_list = glob.glob('{}/*.png'.format(images_path))
    images_list += glob.glob('{}/*.jpg'.format(images_path))
    images_list += glob.glob('{}/*.pickle'.format(images_path))

    return images_list 



def get_info(json_dir = json_dir, json_file = json_file, start = 0, end = 10) :

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


def load_pickle(pickle_path):
    with open(pickle_path, 'rb') as file:
        image = pickle.load(file) 
    return image 


    return sources[start:end], binaries[start: end], instances[start: end] 





def predict( image_paths, path, batch_size = 3):

    x = tf.placeholder(name = 'input', dtype = tf.float32, shape = [None, 256, 512, 3])
    net = LaneNet('test') 
    binary, score = net.inference(x, 'lanenet_model')
    scores = [] 
    binaries = [] 

    with tf.Session() as sess : 
        load_weights(sess, weights_path) 
        
        for i in range(0, len(image_paths), batch_size) :

            batch = image_paths[i: min(i+batch_size, len(image_paths))] 
            images = np.array([normalize(resize_image( read_image(image_path) , (512, 256))) for image_path in batch])
            binary_output, score_output = sess.run([binary, score], {x : images})
            processed_scores= process_score(score_output) 

            for ii, score_result in enumerate(processed_scores) :
                save_image(path, 'score_{}'.format(i+ii), score_result) 
                scores.append(path+'/score_{}'.format(i+ii)+'.png') 

                binary_result = resize_image(np.array(binary_output[ii,:,:]*255, dtype = np.uint8), (1280,720) )
                save_image(path, 'binary_{}'.format(i+ii), binary_result) 
                binaries.append(path+'/binary_{}'.format(i+ii)+'.png') 
    

    return binaries, scores




def process_score(scores) : 

    results = [] 
    for i in range(scores.shape[0]) : 

        score = scores[i, :,:,1]
        score = resize_image(score, (1280, 720)) 
        score[score < 0] = 0 
        score *= (1/np.max(score)) 
        score[score <=0.2] = 0 
        score[score > 0.2] = 255 
        score = np.array(score, dtype = np.uint8) 
        score = open_image(score) 
        results.append(score) 
    
    return results 
    


def open_image(image) :

    idx = image.nonzero()[0] 
    lowest = np.min(idx) 
    separate = np.minimum(lowest + 65 , image.shape[1]-1) 

    kernel11 = np.ones((11,11), np.uint8)
    kernel9 = np.ones((9,9), np.uint8)
    kernel5 = np.ones((5,5), np.uint8) 
    kernel3 = np.ones((3,3), np.uint8) 


    image[lowest:separate//2,:] = cv2.erode(image[lowest:separate//2,:], kernel11, iterations=1) 
    image[separate//2:separate,:] = cv2.erode(image[separate//2:separate,:], kernel9, iterations=1) 

    image = cv2.GaussianBlur(image,(9,9),cv2.BORDER_DEFAULT)
    image[image > 0 ] = 255 
    image = cv2.erode(image, kernel3, iterations=1)
    
    return image






'''
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
'''