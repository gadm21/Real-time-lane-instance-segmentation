
import tensorflow as tf

from config import global_config
import LaneNet_discriminative_loss
from semantic_segmentation_zoo import cnn_basenet

#LaneNet back-end is mainly for binary and instance segmentation loss calculations

CFG= global_config.cfg

class LaneNetBackEnd(cnn_basenet.CNNBaseModel):

    def __init__(self, phase):
        #phase: train or test
        super(LaneNetBackEnd, self).__init__()
        self._phase= phase
        self._is_training= tf.equal(self._phase, tf.constant('train', dtype= tf.string))
    
    @classmethod
    def _compute_class_weighted_cross_entropy_loss(cls, onehot_labels, logits, classes_weights):

        loss_weights= tf.reduce_sum(tf.multiply(onehot_labels, classes_weights), axis=3)
        loss= tf.losses.softmax_cross_entropy(onehot_labels= onehot_labels,logits=logits,weights=loss_weights)

        return loss 
    
    