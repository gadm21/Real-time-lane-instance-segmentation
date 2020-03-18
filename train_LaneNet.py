
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
from evaluate_model import calculate_model_precision, calculate_model_fp, calculate_model_fn, get_image_summary
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
    train_fp= calculate_model_fp(train_prediction_logits, train_binary_labels)
    train_fn= calculate_model_fn(train_prediction_logits, train_binary_labels)


    #summary
    train_binary_seg_ret_for_summary = get_image_summary( img=train_prediction )
    train_embedding_ret_for_summary = get_image_summary( img=train_pix_embedding )

    train_cost_scalar = tf.summary.scalar(
        name='train_cost', tensor=train_total_loss
    )
    train_accuracy_scalar = tf.summary.scalar(
        name='train_accuracy', tensor=train_accuracy
    )
    train_binary_seg_loss_scalar = tf.summary.scalar(
        name='train_binary_seg_loss', tensor=train_binary_seg_loss
    )
    train_instance_seg_loss_scalar = tf.summary.scalar(
        name='train_instance_seg_loss', tensor=train_discriminative_loss
    )
    train_fn_scalar = tf.summary.scalar(
        name='train_fn', tensor=train_fn
    )
    train_fp_scalar = tf.summary.scalar(
        name='train_fp', tensor=train_fp
    )
    train_binary_seg_ret_img = tf.summary.image(
        name='train_binary_seg_ret', tensor=train_binary_seg_ret_for_summary
    )
    train_embedding_feats_ret_img = tf.summary.image(
        name='train_embedding_feats_ret', tensor=train_embedding_ret_for_summary
    )
    train_merge_summary_op = tf.summary.merge(
        [train_accuracy_scalar, train_cost_scalar, train_binary_seg_loss_scalar,
            train_instance_seg_loss_scalar, train_fn_scalar, train_fp_scalar,
            train_binary_seg_ret_img, train_embedding_feats_ret_img]
    )


    # set tensorflow saver
    saver = tf.train.Saver()
    model_save_dir = 'model'
    os.makedirs(model_save_dir, exist_ok=True)
    train_start_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    model_name = 'tusimple_lanenet_{:s}.ckpt'.format( str(train_start_time))
    model_save_path = os.path.join(model_save_dir, model_name)


    #set tensorboard
    tboard_save_path = 'tboard/logs'
    os.makedirs(tboard_save_path, exist_ok=True)



    #set optimizer
    global_step= tf.Variable(0, trainable= False)
    learning_rate= tf.train.polynomial_decay(learning_rate= cfg.TRAIN.LEARNING_RATE,global_step=global_step,decay_steps=cfg.TRAIN.EPOCHS, power= 0.9)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):    
        optimizer= tf.train.MomentumOptimizer(learning_rate= learning_rate, momentum= cfg.TRAIN.MOMENTUM).minimize(loss= train_total_loss, var_list= tf.trainable_variables(), global_step= global_step)

    saver= tf.train.Saver()
    train_epochs= cfg.TRAIN.EPOCHS
    sess= tf.Session()
    summary_writer = tf.summary.FileWriter(tboard_save_path)
    summary_writer.add_graph(sess.graph)

    with sess.as_default():
        
        sess.run(tf.global_variables_initializer())
        train_cost_time_mean= []
        for epoch in range(train_epochs):
            train_start_time= time.time()
            
            try:    
                _, train_c, train_accuracy_figure, lr, train_binary_loss, train_instance_loss,train_embeddings, train_binary_seg_imgs, train_gt_imgs, train_binary_gt_labels, train_instance_gt_labels, train_summary = \
                    sess.run([optimizer, train_total_loss, train_accuracy, learning_rate, train_binary_seg_loss, train_discriminative_loss, train_pix_embedding, train_prediction, train_images, train_binary_labels, train_instance_labels, train_merge_summary_op])            
            except tf.errors.InvalidArgumentError: break

            summary_writer.add_summary(summary=train_summary, global_step=epoch)

            '''
            if epoch % 10 == 0:
                record_training_intermediate_result(
                    gt_images=train_gt_imgs, gt_binary_labels=train_binary_gt_labels,
                    gt_instance_labels=train_instance_gt_labels, binary_seg_images=train_binary_seg_imgs,
                    pix_embeddings=train_embeddings
                )
            '''
            print("accuracy:", train_c)
            print("loss:", train_binary_loss)

        print("saving model")
        saver.save(sess=sess, save_path=model_save_path, global_step=global_step)

    sess.close()


if __name__ == '__main__':
    # init args
    #args = init_args()
    dataset_dir= r"C:\Users\gad\Downloads\Compressed\lanenet-lane-detection\data\data_records"
    train_LaneNet(dataset_dir)