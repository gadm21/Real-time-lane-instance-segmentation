
import numpy as np 
import tensorflow as tf 

from LaneNet.LaneNet_model.LaneNet import LaneNet 





net= LaneNet(phase= "test")
input_tensor= tf.placeholder(name= "input_tensor", shape= [1, 256, 512, 3], dtype= tf.float32)
binary, instance, score= net.inference(input_tensor, name= "lanenet_model")

sess= tf.InteractiveSession() 

weights_path= r"C:\Users\gad\Desktop\data\model_weights\tusimple_lanenet_vgg.ckpt"
saver= tf.train.Saver()
#saver.restore(sess= sess, save_path= weights_path) 


vars = tf.trainable_variables()


print() 
print(vars[0])
print() 
print(vars[1])
print() 
print(vars[len(vars) -1]) 
print() 


















print("done") 
sess.close() 