
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

weights_path = r'C:\Users\gad\Desktop\repos\VOLO\new_weights\tusimple_lanenet_vgg.ckpt'

json_dir = r'C:\Users\gad\Desktop\data\train\clean_start'
json_file = 'label_data_0313.json'

source_dir = 'images/source'
binary_dir = 'images/binary'
instance_dir = 'images/instance' 

def save_instance(dir, name, instance_image):
    instance_path = os.path.join(dir, name) + '.pickle'
    with open(instance_path, 'wb') as file:
        pickle.dump(instance_image, file, protocol=pickle.HIGHEST_PROTOCOL) 


def get_json_info(json_dir, json_file, num_images = 10) :

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




def predict(image_paths):

    if not isinstance(image_paths, list): image_paths = [image_paths] 

    images = np.array([normalize(resize_image( read_image(image_path) , (512, 256))) for image_path in image_paths])
    x = tf.placeholder(name = 'input', dtype = tf.float32, shape = [1, 256, 512, 3])
    net = LaneNet('test') 
    binary_branch = net.inference(x, 'lanenet_model') 

    with tf.Session() as sess :
        load_weights(sess, weights_path) 
        print('\n\n\n') 

        for i, image_path in enumerate(image_paths):
            print("{} processing {}".format(i, image_path)) 

            image = np.array([normalize(resize_image(read_image(image_path), (512, 256)))])
            print("image:{}".format(image.shape))

            binary_output = sess.run(binary_branch, {x : image})
            
            #binary = resize_image(np.array(binary_output[0] * 255, dtype = np.uint8)  , (1280, 720)) 
            

            #show_image(image[0] * 255) 
            print(len(binary_output)) 
            print(binary_output[0].shape) 
            binary = np.array(binary_output[0]*255, dtype = np.uint8) 
            show_image(binary)

            '''
            save_image(source_dir, 'source_{}'.format(i), image[0]) 
            save_image(binary_dir, 'binary_{}'.format(i), binary)
            '''


    


if __name__ == "__main__":

    infos = get_json_info(json_dir, json_file, 1)
    image_paths = [info[0] for info in infos] 
    
    predict(image_paths) 


 


