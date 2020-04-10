
import os
import sys


from LaneNet.semantic_segmentation_zoo import cnn_basenet, vgg16_based_fcn


# this class is mainly used for feature extraction
class LaneNetFrontEnd(cnn_basenet.CNNBaseModel):

    def __init__(self, phase):
        super(LaneNetFrontEnd, self).__init__()
        self._net= vgg16_based_fcn.VGG16FCN(phase= phase)

    def build_model(self, input_tensor, name, reuse):
        return self._net.build_model(input_tensor= input_tensor, name= name, reuse= reuse)

