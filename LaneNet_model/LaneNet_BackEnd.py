
import os
import sys
sys.path.append(os.getcwd())

import tensorflow as tf
from files import global_config
from . import LaneNet_discriminative_loss
from semantic_segmentation_zoo import cnn_basenet

cfg= global_config.cfg

#this class is mainly used for binary and instance segmentation loss computations
class LaneNetBackEnd(cnn_basenet.CNNBaseModel):

    def __init__(self, phase = 'train'):
        super(LaneNetBackEnd, self).__init__()
        self._phase= phase
        self._is_training= phase == 'train'
    

    #classmethod in python is similar to static member function in cpp
    @classmethod
    def _compute_class_weighted_cross_entropy_loss(cls, onehot_labels, logits, classes_weights):
  
        loss_weights = tf.reduce_sum(tf.multiply(onehot_labels, classes_weights), axis=3)

        loss = tf.losses.softmax_cross_entropy(
            onehot_labels=onehot_labels,
            logits=logits,
            weights=loss_weights
        )


        return loss

    #comptue LaneNet loss
    def compute_loss(self, binary_seg_logits, binary_label, name, reuse):
                            

        with tf.variable_scope(name_or_scope= name, reuse= reuse):

            #calculate class weighted binary segmentation loss
            with tf.variable_scope(name_or_scope= 'binary_seg'):
                binary_label_onehot= tf.one_hot(
                    tf.reshape(
                        tf.cast(binary_label, tf.int32),
                        shape= [binary_label.get_shape().as_list()[0],
                                binary_label.get_shape().as_list()[1],
                                binary_label.get_shape().as_list()[2]]),
                        depth= cfg.TRAIN.CLASSES_NUMS,
                        axis= -1
                )
                
                binary_label_plain= tf.reshape( binary_label,
                    shape= [binary_label.get_shape().as_list()[0] *
                            binary_label.get_shape().as_list()[1] *
                            binary_label.get_shape().as_list()[2] *
                            binary_label.get_shape().as_list()[3]]
                )
                
                counts= tf.cast(counts, tf.float32)

                inverse_weights= tf.divide(1.0,tf.log(tf.add(tf.divide(counts, tf.reduce_sum(counts)), tf.constant(1.02))))

                binary_segmentation_loss= self._compute_class_weighted_cross_entropy_loss(
                    onehot_labels= binary_label_onehot,
                    logits= binary_seg_logits,
                    classes_weights= inverse_weights
                )
                


                #total_loss= binary_segmentation_loss + instance_segmentation_loss + l2_reg_loss
                total_loss = binary_segmentation_loss

                ret= {
                    'total_loss': total_loss,
                    'binary_seg_logits': binary_seg_logits,
                    #'instance_seg_logits': pix_embedding,
                    'binary_seg_loss': binary_segmentation_loss,
                    #'discriminative_loss': instance_segmentation_loss 
                }
                
                return ret
            

    def inference(self, binary_seg_logits, name, reuse):  
        
        with tf.variable_scope(name_or_scope=name, reuse=reuse):

            with tf.variable_scope(name_or_scope='binary_seg'):
                binary_seg_score = tf.nn.softmax(logits=binary_seg_logits)
                #binary_seg_score= binary_seg_logits 
                binary_seg_prediction = tf.argmax(binary_seg_score, axis=-1)
            
            '''
            with tf.variable_scope(name_or_scope='instance_seg'):
                pix_bn = self.layerbn(inputdata=instance_seg_logits, is_training=self._is_training, name='pix_bn')
                pix_relu = self.relu(inputdata=pix_bn, name='pix_relu')
                instance_seg_prediction = self.conv2d(
                    inputdata=pix_relu,
                    out_channel=cfg.TRAIN.EMBEDDING_FEATS_DIMS,
                    kernel_size=1,
                    use_bias=False,
                    name='pix_embedding_conv'
                )
            '''
        return binary_seg_prediction, binary_seg_score 


