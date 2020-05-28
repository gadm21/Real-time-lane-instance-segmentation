
import os 
import sys 
sys.path.append(os.getcwd())

import numpy as np 
import cv2 
import pickle 
import tensorflow as tf 
import time     

from LaneNet_model.LaneNet import LaneNet 
from LaneNet_model.LaneNet_PostProcessor import LaneNetPostProcessor
from LaneNet_model.my_postprocessor import * 



def predict( source_image):
    image = normalize(resize_image( source_image , (512, 256)))
    x = tf.placeholder(name = 'input', dtype = tf.float32, shape = [1, 256, 512, 3])
    net = LaneNet('test') 
    binary, instance, binary2= net.inference(x, 'lanenet_model')

    with tf.Session() as sess : 
        start = time.time() 
        binary_image, instance_image, binary_image2 = sess.run([binary, instance, binary2], {x : [image]})
        end = time.time() 

    return {"binary":binary, 'insance':instance, 'binary2':binary2}


def test_postprocessor():

    image = read_image('images/source.jpg') 
    

    ret = predict( image) 
    show_image(ret['binary'][0]) 
    show_image(ret['instance'][0]) 
    show_image(ret['binary2'][0])



if __name__ == "__main__":
    test_postprocessor()
    print("done") 