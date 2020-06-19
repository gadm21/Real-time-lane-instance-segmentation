
import os 
import sys 
sys.path.append(os.getcwd())
import numpy as np 
import tensorflow as tf 

from tensorflow.python.tools import inspect_checkpoint as chkp

weights_path = r'C:\Users\gad\Desktop\repos\VOLO\weights\tusimple_lanenet_vgg.ckpt'



variables = tf.train.list_variables(weights_path) 

from tensorflow.python import pywrap_tensorflow
reader = pywrap_tensorflow.NewCheckpointReader(weights_path)
var_to_shape_map = reader.get_variable_to_shape_map() # 'var_to_shape_map' is a dictionary contains every tensor in the model
 

for v in variables : 
    print(v) 
    