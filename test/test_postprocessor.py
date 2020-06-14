
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

sources_path = 'images/source'
binaries_path = 'images/binary'
instances_path = 'images/instance' 

weights_path = r'C:\Users\gad\Desktop\repos\VOLO\weights\tusimple_lanenet_vgg.ckpt'


def predict( images):
    if not isinstance(images, list): image = [image] 
    images = np.array([normalize(resize_image( image , (512, 256))) for image in images])

    x = tf.placeholder(name = 'input', dtype = tf.float32, shape = [images.shape[0], 256, 512, 3])
    net = LaneNet('test') 
    binary, instance= net.inference(x, 'lanenet_model')

    with tf.Session() as sess : 
        load_weights(sess, weights_path) 
        binary_output, instance_segmentation_output = sess.run([binary, instance], {x : images})

    binary_images = [np.array(binary * 255, dtype = np.uint8) for binary in binary_output] 
    instance_images = [np.array(instance * 255, dtype = np.uint8) for instance in instance_segmentation_output]
    
    return binary_images, instance_images

def test_postprocessor(binary_images):
    if not isinstance(binary_images, list): binary_images = [binary_images]
    postprocessor = PostProcessor() 
    for i, binary_image in enumerate(binary_images):
        
        result = postprocessor.process(binary_image) 
        save_image('images/result', '{}_mypostprocessor_{}_'.format(time.time(), i), result['mask_image'])
        


if __name__ == "__main__":

    source_images_paths = get_image_paths_list(sources_path) 
    binary_images_paths = get_image_paths_list(binaries_path) 
    instance_images_paths = get_image_paths_list(instances_path) 
    source_images_paths, binary_images_paths, instance_images_paths = truncate(source_images_paths,
                                                                                binary_images_paths,
                                                                                instance_images_paths,
                                                                                start = 0,
                                                                                end = 20)
    
    
    postprocessor = PostProcessor() 
    his_postprocessor = LaneNetPostProcessor() 
    my_time = [] 
    his_time = [] 

    for i in range(len(source_images_paths)):
        print("{} processing {}".format(i, source_images_paths[i]))

        source = read_image(source_images_paths[i]) 
        binary = normalize(read_image(binary_images_paths[i]) )
        instance = load_pickle(instance_images_paths[i]) 

        start = time.time() 
        my_ret = postprocessor.process(binary, source) 
        my_time.append(time.time() - start) 

        start = time.time() 
        his_ret = his_postprocessor.postprocess(binary, instance, source) 
        his_time.append(time.time() - start) 
        


    print("my_tim2:{}    |  his_time:{}".format(np.mean(my_time), np.mean(his_time)))

