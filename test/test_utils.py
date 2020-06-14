

import glob 
import tensorflow as tf 
import pickle 
import numpy as np



def load_weights(sess, weights_path):
    saver = tf.train.Saver() 
    saver.restore(sess = sess, save_path= weights_path) 



def get_image_paths_list(images_path):
    images_list = glob.glob('{}/*.png'.format(images_path))
    images_list += glob.glob('{}/*.jpg'.format(images_path))
    images_list += glob.glob('{}/*.pickle'.format(images_path))

    return images_list 

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


def truncate(sources, binaries, instances, start, end) :
    return sources[start:end], binaries[start: end], instances[start: end] 