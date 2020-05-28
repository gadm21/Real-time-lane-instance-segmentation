import argparse
import glob
import time
import json

import cv2
import glog as log
import numpy as np
import tensorflow as tf
#import tqdm

import os
import os.path as ops
import sys
sys.path.append(os.getcwd())

from LaneNet_model import LaneNet 
from LaneNet_model.LaneNet_PostProcessor import LaneNetPostProcessor
from LaneNet_model.my_postprocessor import *
from files import global_config 
CFG= global_config.cfg


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, help='The source tusimple lane test folder. contains json file and images to be evaluated')
    parser.add_argument('--weights_path', type=str, default = r"C:\Users\gad\Desktop\repos\VOLO\weights\tusimple_lanenet_vgg.ckpt", help='The model weights path')
    return parser.parse_args()


def get_json(src):
    for json_file_path in glob.glob('{}/test*.json'.format(src)) :
        return json_file_path

def test_lanenet(args):
    
    src = args.src 
    json_path = get_json(src) 
    avg_time_cost = [] 
    avg_time_hisprocessor = []
    avg_time_myprocessor = []
    counter = 0

    input_tensor = tf.placeholder(name = 'input_tensor', dtype = tf.float32, shape = [1, 256, 512, 3]) 

    net = LaneNet.LaneNet("test")
    postprocessor = LaneNetPostProcessor(ipm_remap_file_path = 'files/tusimple_ipm_remap.yml') 
    advanced_postprocessor = PostProcessor() 
    binary_seg, instance_seg , binary_seg2 = net.inference(input_tensor, name = "lanenet_model")

    with tf.Session() as sess : 
        saver = tf.train.Saver()
        saver.restore(sess = sess, save_path = args.weights_path) 

        with open(json_path, 'r') as file:
            for i, line in enumerate(file):
                info = json.loads(line) 

                image_path = info['raw_file'] 
                image_full_path = ops.join(src, image_path) 
                original_image =resize_image( read_image(image_full_path) , (512, 256))
                image = normalize(original_image)

                start_time = time.time()
                binary_seg_image, instance_seg_image, binary_seg_image2 = sess.run([binary_seg, instance_seg, binary_seg2], {input_tensor: [image]})
                time_cost = time.time() - start_time 
                avg_time_cost.append(time_cost) 

                #binary_seg_image = reverse_normalize(binary_seg_image[0]) 
                #instance_seg_image= reverse_normalize(instance_seg_image[0])
                
                start = time.time()
                ret = postprocessor.postprocess(binary_seg_image[0], instance_seg_image[0], original_image) 
                checkpoint = time.time()
                my_ret = advanced_postprocessor.process(binary_seg_image[0]) 
                end = time.time()
                avg_time_hisprocessor.append(checkpoint - start) 
                avg_time_myprocessor.append(end - checkpoint ) 


                '''
                save_image('images', 'source_image_{}'.format(i) , ret['source_image'])
                save_image('images', 'his_mask_image_{}'.format(i), ret['mask_image'])
                save_image('images', 'my_mask_image_{}'.format(i), my_ret['mask_image'])
                


                for lane in ret['fit_params'] :
                    print(lane) 
                print("___________________________")
                for lane in my_ret['fit_params']:
                    print(lane) 
                print('\n\n')
                '''
                print("his shape:", ret['mask_image'].shape) 
                print("my shape:", my_ret['mask_image'].shape) 
                
                return 
                counter += 1
                if counter % 15 == 0 : 
                    print("average processing time:{:.5f} s".format( np.mean(avg_time_cost)))
                    print("average his postprocessing time:{:.5f} s".format( np.mean(avg_time_hisprocessor)))
                    print("average my postprocessing time:{:.5f} s".format( np.mean(avg_time_myprocessor)))
                    return
                



if __name__ == "__main__":
    
    args = init_args() 
    
    test_lanenet(args) 

