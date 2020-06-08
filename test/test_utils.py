

import glob 
import tensorflow as tf 

def get_image_paths_list(images_path):
    images_list = glob.glob('{}/*.png'.format(images_path))
    images_list += glob.glob('{}/*.jpg'.format(images_path))
    return images_list 

def load_weights(sess, weights_path):
    saver = tf.train.Saver() 
    saver.restore(sess = sess, save_path= weights_path) 