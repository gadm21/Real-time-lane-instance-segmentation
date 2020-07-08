

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

json_dir = r'C:\Users\gad\Desktop\data\train\clean_start\test'
json_file = 'test_tasks_0627.json'

label_json_dir = r'C:\Users\gad\Desktop\data' 
label_json_file = 'test_label.json'

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
        coords_x = np.int_(lane_params[0]*coords_y**3+lane_params[1]*coords_y**2 + lane_params[2]*coords_y + lane_params[3])
        #coords_x[coords_x<0] = -2 
        #coords_x[coords_x>1179] = -2
        lanes.append(coords_x)
    
    lanes = order_lanes(lanes) 
    lanes[lanes<0] = -2 
    lanes[lanes>1179] = -2 
    lanes = lanes.tolist()

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





def show(image, h, mine, his, id, save = 'mine_vsgt', display= True):
   
    for lane in his :
        for i in range(len(lane)):
            if int(lane[i]) == -2 : continue 
            cv2.circle(image, (int(lane[i]), int(h[i])), thickness = 2, color = [0,255,0], radius = 2) 
    
    for lane in mine : 
        for i in range(len(lane)):
            if int(lane[i]) == -2 : continue 
            cv2.circle(image,  (int(lane[i]), int(h[i])), thickness = 2, color = [255,0,0], radius = 2) 
    
    if display : show_image(image) 
    if save : 
        save_image(save, str(id), image) 
   
    



def run() :     

    lines = get_json_lines(json_dir, json_file) 
    gt_lines = get_json_lines(label_json_dir, label_json_file) 

    h = lines[0]['h_samples'] 
    pp = PostProcessor() 

    input = tf.placeholder(dtype = tf.float32, shape=[None, 256,512,3]) 
    net = LaneNet() 
    b, s = net.inference(input) 

    with open('pred_file.json', 'w') as file : 
        with tf.Session() as sess : 
            load_weights(sess, weights_path)

            counter = 0
            start = time.time()
            for line in lines : 
                print("line:", counter) 
                counter += 1

                image_path = ops.join(json_dir, line['raw_file']) 
                image = np.array([normalize(resize_image( read_image(image_path) , (512, 256)))])

                binary, score = sess.run([b,s], {input:image}) 
                binary_image = process_binary(denormalize(binary[0]) ) 
            
                ret = pp.process(binary_image) 
                lanes = ret['rightful_lanes_points']#get_lanes(h,ret["lanes_params"])
                line['lanes'] = lanes 
                file.write(json.dumps(line)+'\n')
                show(read_image(image_path), h, lanes, gt_lines[counter-1]['lanes'], id=counter, display = False) 
                if counter == 100 : break 

            print("time:", time.time() - start) 
    


run() 

'''
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
            