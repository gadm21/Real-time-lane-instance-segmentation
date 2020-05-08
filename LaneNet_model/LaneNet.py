import tensorflow as tf 

from files import global_config
from . import LaneNet_BackEnd, LaneNet_FrontEnd
from semantic_segmentation_zoo import cnn_basenet

cfg= global_config.cfg



class LaneNet(cnn_basenet.CNNBaseModel):
    
    def __init__(self, phase, reuse= False):
        super(LaneNet, self).__init__()
        self._net_flag= 'vgg'
        self._reuse= reuse

        self._frontend= LaneNet_FrontEnd.LaneNetFrontEnd(phase= phase)
        self._backend= LaneNet_BackEnd.LaneNetBackEnd(phase= phase)

    
    def inference(self, input_tensor, name):

        with tf.variable_scope(name_or_scope= name, reuse= self._reuse):
            #first, extract image feautres
            layers_outputs= self._frontend.build_model(input_tensor= input_tensor, name= 'vgg_frontend', reuse= self._reuse)

            #second, apply backend process
            binary_seg_prediction, instance_seg_prediction, binary_score= self._backend.inference(
                binary_seg_logits= layers_outputs['binary_segment_logits']['data'],
                instance_seg_logits= layers_outputs['instance_segment_logits']['data'],
                name= 'vgg_backend', reuse= self._reuse
            )

            self._reuse= True
        
        return binary_seg_prediction, instance_seg_prediction, binary_score
    


    def compute_loss(self, input_tensor, binary_label, instance_label, name):

        with tf.variable_scope(name_or_scope= name, reuse= self._reuse):
            #first, extract image features
            layers_outputs= self._frontend.build_model(input_tensor= input_tensor, name= 'vgg_frontend', reuse= self._reuse)
            
            #binary segment logits shape: (?, 256, 512, 2)
            #instance segmnet logits shape: (?, 256, 512, 64)

            #binary segment labels shape: (?, 256, 512, 1)
            #instance segment labels shape: (?, 256, 512, 1)

            #second, apply backend process
            calculated_losses= self._backend.compute_loss(
                binary_seg_logits= layers_outputs['binary_segment_logits']['data'],
                binary_label= binary_label,
                instance_seg_logits= layers_outputs['instance_segment_logits']['data'],
                instance_label= instance_label,
                name= 'vgg_backend',
                reuse= self._reuse
            )

            self._reuse= True
        
        return calculated_losses
