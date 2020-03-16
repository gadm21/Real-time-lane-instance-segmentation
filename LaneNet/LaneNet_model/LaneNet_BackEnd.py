
import os
import sys
sys.path.append(os.getcwd())

import tensorflow as tf
import global_config
from . import LaneNet_discriminative_loss
from LaneNet.semantic_segmentation_zoo import cnn_basenet

cfg= global_config.cfg

#this class is mainly used for binary and instance segmentation loss computations
class LaneNetBackEnd(cnn_basenet.CNNBaseModel):

    def __init__(self, phase):
        super(LaneNetBackEnd, self).__init__()
        self._phase= phase
        self._is_training= self.check_if_net_for_training()
    
    def check_if_net_for_training(self):

        if isinstance(self._phase, tf.Tensor):
            phase= self._phase
        else:
            phase= tf.constant(self._phase, dtype= tf.string)
        
        return tf.equal(phase, tf.constant('train', dtype= tf.string))
    

    #classmethod in python is similar to static member function in cpp
    @classmethod
    def _compute_class_weighted_cross_entropy_loss(cls, onehot_labels, logits, classes_weights):
        """

        :param onehot_labels:
        :param logits:
        :param classes_weights:
        :return:
        """
        loss_weights = tf.reduce_sum(tf.multiply(onehot_labels, classes_weights), axis=3)

        loss = tf.losses.softmax_cross_entropy(
            onehot_labels=onehot_labels,
            logits=logits,
            weights=loss_weights
        )

        print()
        print()
        print("compute class weighted cross entropy")
        print("classes weighgts shape:", classes_weights.shape.as_list())
        print("loss_weights shape:", loss_weights.shape.as_list())
        print("loss shape:", loss.shape.as_list())
        print()
        print() 

        return loss

    #comptue LaneNet loss
    def compute_loss(self, binary_seg_logits, binary_label, 
                            instance_seg_logits, instance_label,
                            name, reuse):
        
        print()
        print()
        print()
        print("part I")
        print("binary_seg_logits:", type(binary_seg_logits), " ", binary_seg_logits.shape)
        print("instance_seg_logits:", type(instance_seg_logits), " ", instance_seg_logits.shape)
        print("binary_label:", type(binary_label), " ", binary_label.shape)
        print("instance_label:", type(instance_label), " ", instance_label.shape)

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

                print("binary_label_onehot:", type(binary_label_onehot), " ", binary_label_onehot.get_shape().as_list())
                print("TRAIN.CLASSES_NUM:", cfg.TRAIN.CLASSES_NUMS)

                binary_label_plain= tf.reshape(
                    binary_label,
                    shape= [binary_label.get_shape().as_list()[0] *
                            binary_label.get_shape().as_list()[1] *
                            binary_label.get_shape().as_list()[2] *
                            binary_label.get_shape().as_list()[3]]
                )
                print("binary_label_plain:", type(binary_label_plain), " ", binary_label_plain.get_shape().as_list())

                unique_labels, unique_ids, counts= tf.unique_with_counts(binary_label_plain)
                print("unique_labels:", type(unique_labels), " ", unique_labels.get_shape().as_list())
                print("unique_ids:", type(unique_ids), " ", unique_ids.get_shape().as_list())
                print("counts:", type(counts), " ", counts.get_shape().as_list())
                print()
                

                counts= tf.cast(counts, tf.float32)

                inverse_weights= tf.divide(
                    1.0,
                    tf.log(tf.add(tf.divide(counts, tf.reduce_sum(counts)), tf.constant(1.02)))
                )

                binary_segmentation_loss= self._compute_class_weighted_cross_entropy_loss(
                    onehot_labels= binary_label_onehot,
                    logits= binary_seg_logits,
                    classes_weights= inverse_weights
                )

            #________________________________________________________________________________
            #________________________________________________________________________________
            #________________________________________________________________________________
            
            

            #calculate class weighted instance segmentation loss
            with tf.variable_scope(name_or_scope= 'instance_seg'):

                pix_bn= self.layerbn(inputdata= instance_seg_logits, is_training= self._is_training, name= 'pix_bn')
                pix_relu= self.relu(inputdata= pix_bn, name='pix_relu')
                pix_embedding= self.conv2d(
                    inputdata= pix_relu,
                    out_channel= cfg.TRAIN.EMBEDDING_FEATS_DIMS,
                    kernel_size= 1,
                    use_bias= False,
                    name= 'pix_embedding_conv'
                )
                pix_image_shape= (pix_embedding.get_shape().as_list()[1], pix_embedding.get_shape().as_list()[2])
                
                
                print() 
                print() 
                print() 
                print("part II")
                print("pix embedding shape:", pix_embedding.shape.as_list())
                print("instance label shape:", instance_label.shape.as_list())
                print() 

                instance_segmentation_loss, Lvar, Ldist, Lreg= \
                    LaneNet_discriminative_loss.discriminative_loss(
                        pix_embedding, instance_label, cfg.TRAIN.EMBEDDING_FEATS_DIMS,
                        pix_image_shape, 0.5, 3.0, 1.0, 1.0, 0.001
                    )
                
                l2_reg_loss= tf.constant(0.0, tf.float32)
                for vv in tf.trainable_variables():
                    if 'bn' in vv.name or 'gn' in vv.name:
                        continue
                    else:
                        l2_reg_loss= tf.add(l2_reg_loss, tf.nn.l2_loss(vv))
                l2_reg_loss *= 0.001




                total_loss= binary_segmentation_loss + instance_segmentation_loss + l2_reg_loss

                ret= {
                    'total_loss': total_loss,
                    'binary_seg_logits': binary_seg_logits,
                    'instance_seg_logits': pix_embedding,
                    'binary_seg_loss': binary_segmentation_loss,
                    'discriminative_loss': instance_segmentation_loss 
                }

                print() 
                print() 
                print() 
                return ret


    def inference(self, binary_seg_logits, instance_seg_logits, name, reuse):
        """

        :param binary_seg_logits:
        :param instance_seg_logits:
        :param name:
        :param reuse:
        :return:
        """
        with tf.variable_scope(name_or_scope=name, reuse=reuse):

            with tf.variable_scope(name_or_scope='binary_seg'):
                binary_seg_score = tf.nn.softmax(logits=binary_seg_logits)
                binary_seg_prediction = tf.argmax(binary_seg_score, axis=-1)

            with tf.variable_scope(name_or_scope='instance_seg'):

                pix_bn = self.layerbn(
                    inputdata=instance_seg_logits, is_training=self._is_training, name='pix_bn')
                pix_relu = self.relu(inputdata=pix_bn, name='pix_relu')
                instance_seg_prediction = self.conv2d(
                    inputdata=pix_relu,
                    out_channel=CFG.TRAIN.EMBEDDING_FEATS_DIMS,
                    kernel_size=1,
                    use_bias=False,
                    name='pix_embedding_conv'
                )

        return binary_seg_prediction, instance_seg_prediction