
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



def predict(net, source_image):
    image = normalize(source_image) 
    x = tf.placeholder(name = 'input', dtype = tf.float32, shape = [1, 256, 512, 3])
    binary, instance, binary2= net.inference(x, 'lanenet_model')

    start = time.time() 
    binary_image, instance_image, binary_image2 = sess.run([binary, instance, binary2], feed_dict={x:[image]})
    end = time.time() 

    return {"binary":binary, 'insance':instance, 'binary2':binary2}

def test_postprocessor():

    image = read_image('images/source.jpg') 
    net = LaneNet() 

    ret = prdict(net, image) 
    show_image(ret['binary'][0]) 
    show_image(ret['instance'][0]) 
    show_image(ret['binary2'][0])



if __name__ == "__main__":
    test_postprocessor()
    print("done") 