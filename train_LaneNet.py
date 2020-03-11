
import argparse
import math
import time
import cv2
import glog as log
import numpy as np
import tensorflow as tf

import os
import sys
sys.path.append(os.getcwd())

from LaneNet.data_provider import LaneNet_data_feed_pipeline
from LaneNet.LaneNet_model import LaneNet
from evaluate_model import calculate_model_precision, calculate_model_fp, calculate_model_fn
import global_config
cfg= global_config.cfg 



def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--dataset_dir', type=str,
                        help='Lanenet Dataset dir')
    parser.add_argument('-w', '--weights_path', type=str, default= None,
                        help='Path to pre-trained weights to continue training')
    parser.add_argument('-m', '--multi_gpus', type=args_str2bool, default=False,
                        nargs='?', const=True, help='Use multi gpus to train')
    parser.add_argument('--net_flag', type=str, default='vgg',
                        help='The net flag which determins the net\'s architecture')

    return parser.parse_args()






def train_LaneNet (dataset_dir, weights_path= None):

 
    train_dataset= LaneNet_data_feed_pipeline.LaneNetDataFeeder(dataset_dir, flags= 'train')
    val_dataset= LaneNet_data_feed_pipeline.LaneNetDataFeeder(dataset_dir, flags= 'val')
    


    train_net= LaneNet.LaneNet(phase= 'train', reuse= False)
    val_net= LaneNet.LaneNet(phase= 'val', reuse= True)


    train_images, train_binary_labels, train_instance_labels= train_dataset.inputs(cfg.TRAIN.BATCH_SIZE, 1)
    
    print("train_images:", train_images.shape, "...................................")
    print("binary_labels:", train_binary_labels.shape, "...........................")
    print("instance_labels:", train_instance_labels.shape, ".......................")
    
    #compute loss 
    train_compute_ret= train_net.compute_loss(input_tensor= train_images, 
                                            binary_label= train_binary_labels,
                                            instance_label= train_instance_labels,
                                            name= 'lanenet_model')

    train_total_loss= train_compute_ret['total_loss']
    train_binary_seg_loss= train_compute_ret['binary_seg_loss']
    train_discriminative_loss= train_compute_ret['discriminative_loss']


    #compute accuracy
    train_pix_embedding= train_compute_ret['instance_seg_logits']    
    train_prediction_logits= train_compute_ret['binary_seg_logits']
    train_prediction_score= tf.nn.softmax(logits= train_prediction_logits)
    train_prediction= tf.argmax(train_prediction_score, axis= -1)
    train_accuracy= calculate_model_precision(train_prediction_logits, train_binary_labels)
    train_false_positive= calculate_model_fp(train_prediction_logits, train_binary_labels)
    train_false_negative= calculate_model_fn(train_prediction_logits, train_binary_labels)

    #set optimizer
    global_step= tf.Variable(0, trainable= False)
    learning_rate= tf.train.polynomial_decay(learning_rate= cfg.TRAIN.LEARNING_RATE,global_step=global_step,decay_steps=cfg.TRAIN.EPOCHS, power= 0.9)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):    
        optimizer= tf.train.MomentumOptimizer(learning_rate= learning_rate, momentum= cfg.TRAIN.MOMENTUM).minimize(loss= train_total_loss, var_list= tf.trainable_variables(), global_step= global_step)

    saver= tf.train.Saver()
    train_epochs= cfg.TRAIN.EPOCHS
    sess= tf.Session()
    with sess.as_default():
        
        sess.run(tf.global_variables_initializer())
        train_cost_time_mean= []
        for epoch in range(train_epochs):
            train_start_time= time.time()
            
            try:    
                _, train_c, train_accuracy_figure, lr, train_binary_loss, train_instance_loss,train_embeddings, train_binary_seg_imgs, train_gt_imgs, train_binary_gt_labels, train_instance_gt_labels = sess.run([optimizer, train_total_loss, train_accuracy, learning_rate, train_binary_seg_loss, train_discriminative_loss, train_pix_embedding, train_prediction, train_images, train_binary_labels, train_instance_labels])            
            except tf.errors.InvalidArgumentError: break

            print("accuracy:", train_c)
            print("loss:", train_binary_loss)


if __name__ == '__main__':
    # init args
    #args = init_args()
    dataset_dir= r"C:\Users\gad\Downloads\Compressed\lanenet-lane-detection\data\data_records"
    train_LaneNet(dataset_dir)