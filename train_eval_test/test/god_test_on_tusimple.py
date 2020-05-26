import argparse
import glob
import time

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
    """
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, help='The source tusimple lane test folder. contains json file and images to be evaluated')
    parser.add_argument('--weights_path', type=str, help='The model weights path')
    return parser.parse_args()


def get_json(src):
    for json_file_path in glob.glob('{}/test*.json'.format(src)) :
        return json_file_path

def test_lanenet(args):

    weights = args.weights_path 
    json_file = args.json_file 

    

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

    with sess.as_default():

        saver.restore(sess=sess, save_path=weights_path)

        image_list = glob.glob('{:s}/**/*.jpg'.format(src_dir), recursive=True)
        avg_time_cost = []
        for index, image_path in tqdm.tqdm(enumerate(image_list), total=len(image_list)):

            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            image_vis = image
            image = cv2.resize(image, (512, 256), interpolation=cv2.INTER_LINEAR)
            image = image / 127.5 - 1.0

            t_start = time.time()
            binary_seg_image, instance_seg_image = sess.run(
                [binary_seg_ret, instance_seg_ret],
                feed_dict={input_tensor: [image]}
            )
            avg_time_cost.append(time.time() - t_start)

            postprocess_result = postprocessor.postprocess(
                binary_seg_result=binary_seg_image[0],
                instance_seg_result=instance_seg_image[0],
                source_image=image_vis
            )

            if index % 100 == 0:
                log.info('Mean inference time every single image: {:.5f}s'.format(np.mean(avg_time_cost)))
                avg_time_cost.clear()

            input_image_dir = ops.split(image_path.split('clips')[1])[0][1:]
            input_image_name = ops.split(image_path)[1]
            output_image_dir = ops.join(save_dir, input_image_dir)
            os.makedirs(output_image_dir, exist_ok=True)
            output_image_path = ops.join(output_image_dir, input_image_name)
            if ops.exists(output_image_path):
                continue

            cv2.imwrite(output_image_path, postprocess_result['source_image'])

    return



if __name__ == "__main__":
    
    src = input("src") 
    get_json(src) 

