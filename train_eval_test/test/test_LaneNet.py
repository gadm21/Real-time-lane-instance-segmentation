
import pickle
import numpy as np 
import tensorflow as tf 
import os 
import sys
import cv2
sys.path.append(os.getcwd())
import argparse

from LaneNet_model import LaneNet 
from LaneNet_model.LaneNet_PostProcessor import LaneNetPostProcessor
from LaneNet_model.my_postprocessor import *
from files import global_config 
cfg= global_config.cfg


def init_args():

    parser= argparse.ArgumentParser()
    parser.add_argument("--image", dest = "image", type= str)
    parser.add_argument("--weights", dest = "weights", type= str)
    return parser.parse_args() 

def minmax_scale(instance_seg_image):
    #instance_seg_image shape: (512, 256, 4)
    #convert range from [-1, 1] back to [0, 255]

    #return (instance_seg_image + 1.0) * 127.5

    for channel in range(cfg.TRAIN.EMBEDDING_FEATS_DIMS):
        current_channel= instance_seg_image[:, :, channel]
        min_value= np.min(current_channel)
        max_value= np.max(current_channel)

        instance_seg_image[:, :, channel]= (current_channel - min_value) * 255 / ( max_value - min_value) 
    
    return instance_seg_image




def get_lanes_masks (image, weights_path):
    print("getting lanes binary & instance masks ...", end = " ")
    
    image = resize_image(image, (512, 256))
    image = normalize(image) 
    input_tensor = tf.placeholder(name = 'input_tensor', dtype = tf.float32, shape = [1, 256, 512, 3]) 

    net = LaneNet.LaneNet("test")
    binary_seg, instance_seg , binary_seg2 = net.inference(input_tensor, name = "lanenet_model")

    with tf.Session() as sess : 
        saver = tf.train.Saver()
        saver.restore(sess = sess, save_path = weights_path) 

        binary_seg_image, instance_seg_image, binary_seg_image2 = sess.run([binary_seg, instance_seg, binary_seg2], {input_tensor: [image]})

    binary_seg_image = reverse_normalize(binary_seg_image[0]) 
    instance_seg_image= reverse_normalize(instance_seg_image[0])
    print("DONE")
    return binary_seg_image, instance_seg_image



def get_lane_curves(binary_image):
    print("getting lanes curves...", end= "")
    postprocessor = PostProcessor() 
    clusters = postprocessor.process(binary_image) 

    lanes = []
    for cluster in clusters:
        if cluster.valid : lanes.append(cluster)
    
    
    lanes.sort(key = lambda lane : lane.means[0][1], reverse = True) 

    lane_curves = []
    start_points = []
    for lane in lanes:
        lane_curves.append(lane.get_curve()) 
        start_points.append(lane.get_start_point())
    
    print("DONE")
    return lane_curves , start_points 


def draw_curve(image, curve, color, start_from =10):
    print("drawing curves...", end= " ")
    image= to_colored(image) 

    
    def draw_lane(image, y, x, color= [255, 0, 0], thickness= 5):

        points = len(x)
        for i, _ in enumerate(x):
            new_thickness = int(thickness * (i/ points))
            cv2.circle(image, (x[i], y[i]), new_thickness,  color, -1)

        return image 

    end_point = image.shape[0] - 10
    step = (image.shape[0] -10 - start_from)/2
    y = np.linspace(start_from, end_point, step, endpoint= False).astype(np.int32)
    x = np.array(curve[0]*y**2 + curve[1]*y + curve[2], dtype = np.int32)
    
    image = draw_lane(image, y, x, color = color) 
    
    print("DONE")
    return image 


def colorize_lanes(binary):

    p = PostProcessor()
    lanes = p.process(binary) 
    
    color_map= [[0, 255, 0], [0, 0, 255], [255, 0, 0]]

    for i, lane in enumerate(lanes):
        c = color_map[i % len(color_map)]
        binary = lane.colorize(binary, c, False) 
    
    return binary 

def run(args):
    image = read_image(args.image) 
    original_shape = image.shape
    image = resize_image(image , (512, 256))
    

    binary, instance = get_lanes_masks(image, args.weights)
    colored_binary = colorize_lanes(binary) 
    save_image("iamges/results", "binary", colored_binary)
    show_image(colored_binary) 
    '''
    lane_curves, start_points = get_lane_curves(binary)

    color_map= [[0, 255, 0], [0, 0, 255], [255, 0, 0]]
    for i, curve in enumerate(lane_curves):
        image = draw_curve(image, curve, color_map[i % len(color_map)], start_points[i])
        


    image = resize_image(image, (original_shape[1], original_shape[0])) 
    show_image(image) 
    '''




if __name__ == "__main__":
    args = init_args()
    
    
    run(args) 
    