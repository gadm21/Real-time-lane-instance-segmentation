

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


'''
delete the instance segmentation branch, restore only relevant weights from total weights (from weights_path) 
    and save these weights back to a new file (to new_weights_path) 
'''

weights_path = r'C:\Users\gad\Desktop\repos\VOLO\weights\tusimple_lanenet_vgg.ckpt'
new_weights_path = r'C:\Users\gad\Desktop\repos\VOLO\new_weights\tusimple_lanenet_vgg.ckpt'


def check(name) :
    if 'instance' in name : return False 
    else : return True 


def run():
    
    net = LaneNet('test') 
    x = tf.placeholder(name = 'input', dtype = tf.float32, shape = [1, 256, 512, 3])
    binary = net.inference(x, 'lanenet_model') 

    with tf.Session() as sess :

        restore_vars = [v for v in tf.global_variables() if check(v.name)]
        restore_vars_dict = {} 

        for v in restore_vars : restore_vars_dict[v.name[:-2]] = v 
        saver = tf.train.Saver(restore_vars_dict) 
        saver.restore(sess = sess, save_path = weights_path) 

        saver.save(sess, new_weights_path)
    


         
    print("done") 



run() 