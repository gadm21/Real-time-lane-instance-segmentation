
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

sources_path = 'sources'
binaries_path = 'images/binary'
instances_path = 'images/instance' 

weights_path = r'C:\Users\gad\Desktop\repos\VOLO\weights\tusimple_lanenet_vgg.ckpt'




if __name__ == "__main__":

    source_images_paths = get_image_paths_list(sources_path) 
    
    x = tf.placeholder(shape = [None,256,512,3], type = tf.float32) 
    net = LaneNet() 
    b,s = net.inference() 
    
    
    postprocessor = PostProcessor() 
    his_postprocessor = LaneNetPostProcessor() 

    my_forward_time = [] 
    his_forward_time = []
    my_postprocessing_time = [] 
    his_postprocessing_time = [] 

    with tf.Session() as sess : 
      load_weights(sess, weights_path) 
      for i in range(len(source_images_paths)):
          print("{} processing {}".format(i, source_images_paths[i]))

          source = np.array([normalize(resize_image(read_image(source_images_paths[i]), (512,256)))])
          start = time.time() 
          binary, score = sess.run([b,s], {x:source})
          forward_time = time.time() - start 

          binary_image = denormalize(binary[0]) 
          start = time.time() 
          my_ret = postprocessor.process(binary_image) 
          processing_time = time.time() - start 

          my_forward_time.append(forward_time) 
          my_postprocessing_time.append(processing_time) 
        


      print("my forward time:{} | my postprocessing time:{}".format(np.mean(my_forward_time), np.mean(my_postprocessing_time)))

