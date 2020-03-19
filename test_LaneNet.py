

import numpy as np 
import tensorflow as tf 
import os 
import sys
import cv2
sys.path.append(os.getcwd())
import argparse

from LaneNet.LaneNet_model import LaneNet 
from LaneNet.LaneNet_model.LaneNet_PostProcessor import LaneNetPostProcessor
import global_config 
cfg= global_config.cfg


def init_args():

    parser= argparse.ArgumentParser()
    parser.add_argument("--image_path", type= str)
    parser.add_argument("--weights_path", type= str)

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

def test_LaneNet(image_path, weights_path):

    assert os.path.exists(image_path), "{:s} doesnot exist".format(image_path) 

    image= cv2.imread(image_path, cv2.IMREAD_COLOR)
    original_copy= image
    image= cv2.resize(image, (512, 256), interpolation= cv2.INTER_LINEAR)
    image = image / 127.5 - 1.0 #normalize
    print("image loaded")

    input_tensor= tf.placeholder(name= "input_tensor", dtype= tf.float32, shape= [1, 256, 512, 3])

    net= LaneNet.LaneNet("test")
    binary_seg, instance_seg, binary_score= net.inference(input_tensor, name= "lanenet_model")

    saver= tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess= sess, save_path= weights_path) 

        binary_seg_image, instance_seg_image, binary_score_image=\
            sess.run([binary_seg, instance_seg, binary_score], \
                    feed_dict= {input_tensor: [image]})

        binary_seg_image= binary_seg_image[0]
        instance_seg_image= instance_seg_image[0]
    print("result inferred")

    postprocessor = LaneNetPostProcessor()
    postprocessor_result= postprocessor.postprocess(
        source_image= original_copy,
        binary_seg_result = binary_seg_image,
        instance_seg_result= instance_seg_image
    )
    mask_image= postprocessor_result["mask_image"]
    final_result= postprocessor_result["source_image"]

    instance_seg_image= np.array( minmax_scale(instance_seg_image), np.uint8)


    cv2.imwrite("binary_seg_image.png", binary_seg_image * 255)
    cv2.imwrite("instance_seg_image.png", instance_seg_image)
    cv2.imwrite("mask_image.png", mask_image) 
    cv2.imwrite("final_result.png", final_result)
    print("results saved")

if __name__ == "__main__":

    args= init_args() 

    test_LaneNet(image_path= args.image_path, weights_path= args.weights_path)