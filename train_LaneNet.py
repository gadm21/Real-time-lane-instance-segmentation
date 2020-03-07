
import argparse
import math
import time
import cv2
import glog as log
import numpy as np
import tensorflow.compat.v1 as tf

import os
import sys
sys.path.append(os.getcwd())

from LaneNet.data_provider import LaneNet_data_feed_pipeline

import global_config
cfg= global_config.cfg 




print("done")


def train_LaneNet (dataset_dir, weights_path):

 
    train_dataset= LaneNet_data_feed_pipeline.LaneNetDataFeeder(dataset_dir, flags= 'train')
    val_dataset= LaneNet_data_feed_pipeline.LaneNetDataFeeder(dataset_dir, flags= 'val')
    


    train_net= LaneNet.LaneNet(phase= 'train', reuse= False)
    val_net= LaneNet.LaneNet(phase= 'val', reuse= True)


    train_images, train_binary_labels, train_instance_labels= train_dataset.inputs(cfg.TRAIN.BATCH_SIZE, 1)
    
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
    train_accuracy= evaluate_model_utils.calculate_model_precision(train_prediction_logits, train_binary_labels)
    train_false_positive= evaluate_model_utils.calculate_model_fp(train_prediction_logits, train_binary_labels)
    train_false_negative= evaluate_model_utils.calculate_model_fn(train_prediction_logits, train_binary_labels)

    #set optimizer
    global_step= tf.Variable(0, trainable= False)
    learning_rate= tf.train.polynomial_decay(learning_rate= cfg.TRAIN.LEARNING_RATE,global_step=global_step,decay_steps=cfg.TRAIN.EPOCHS, power= 0.9)
    optimizer= tf.train.MomentumOptimizer(learning_rate= learning_rate, momentum= cfg.TRAIN.MOMENTUM).minimize(loss= train_total_loss, var_list= tf.trainable_variables, global_step= global_step)
    
    saver= tf.train.Saver()
    train_epochs= cfg.TRAIN_EPOCHS
    sess= tf.Session()
    with sess.as_default():
        
        train_cost_time_mean= []
        for epoch in rangte(train_epochs):
            train_start_time= time.time()

            _, train_C, train_accuracy_figure,
            lr, train_summary, train_binary_loss, train_instance_loss,\
            train_embeddings, train_binary_seg_images, train_gt_images,\
            train_binary_gt_labels, train_instance_gt_labels = sess.run([optimizer, train_total_loss, train_accuracy, learning_rate])

            print("accuracy:", train_c)
            print("loss:", train_binary_loss)