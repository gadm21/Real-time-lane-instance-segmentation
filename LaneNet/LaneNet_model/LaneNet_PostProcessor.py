

import math
import os 
import cv2 
import glog as log
import numpy as np
#density based spatial clustering of applications with noise 
from sklearn.cluster import DBSCAN
#standardize features by removing the mean and scaling to unit variance
from sklearn.preprocessing import StandardScaler

import global_config 
cfg= global_config.cfg 


#morphological process fills the holes in the binary segmentation result
def morphological_process(image, kernel_size= 5):
    assert (len(image.shape) == 3), "binary image must have a single channel"
    image= np.array(image, np.uint8) 

    kernel= cv2.getStructuringElement(shape= cv2.MORPH_ELLIPSE, ksize= (kernel_size, kernel_size))
    filled_holes= cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=1 )

    return filled_holes 

#converts an image to grayscale 
def toGray(image):
    if len(image.shape == 3):
        gray_image= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else: gray_image= image

    return gray_image

#remove small components
def connect_components_analysis(image):
    gray_image= toGray(image)
    return cv2.connectedComponentsWithStats(gray_image, connectivity= 8, ltype= cv2.CV_32S)


class Lane(object):

    def __init__(self, feat, coord, class_id= -1):
        self._feat= feat
        self._coord= coord 
        self._class_id= class_id 
    
    @property
    def feat(self):
        return self._feat
    
    @property
    def coord(self):
        return self._coord 
    
    @property 
    def class_id(self):
        return self._class_id 
    
    @feat.setter
    def feat(self, value):
        if not isinstance(value, np.ndarray) or value.dtype != np.float64:
            value= np.array(value, np.float64)
        self._feat= value
    
    @coord.setter
    def coord(self, value):
        if not isinstance(value, np.ndarray) or value.dtype != np.int32:
            value= np.array(value, np.int32) 
        self._coord= value
    
    @class_id.setter    
    def class_id(self, value):
        assert isinstance(value, np.int64), "class id must be integer"
        self._class_id= value
    


class LaneCluster(object):

    def __init__(self):
        self._color_map = [np.array([255, 0, 0]),
                    np.array([0, 255, 0]),
                    np.array([0, 0, 255]),
                    np.array([125, 125, 0]),
                    np.array([0, 125, 125]),
                    np.array([125, 0, 125]),
                    np.array([50, 100, 50]),
                    np.array([100, 50, 100])]
    
    @staticmethod
    def get_lane_embedding_features(binary, instance):
    #get lane embedding features according to binary

        #lane_embedding_features are the values in the instance_seg_image on the lanes coords
        idx= np.where(binary==255) 
        lane_embedding_features= instance[idx]
        lane_coordinates= np.vstack((idx[1], idx[0])).transpose()
        return lane_embedding_features, lane_coordinates
        
    @staticmethod
    def embedding_features_dbscan_cluster(lane_embedding_features):
        
        db= DBSCAN(cfg.POSTPROCESS.DBSCAN_EPS , cfg.POSTPROCESS.DBSCAN_MIN_SAMPLES) 

        try:
            features= StandardScaler().fit_transform(lane_embedding_features)
            db.fit(features)
        except Exception as e:
            print("clustering error: {:s}".format(e))
            return None, None, None, None
        
        db_labels= db.labels_
        unique_labels= np.unique(dp_labels) 
        num_clusters= len(unique_labels)
        cluster_centers= db.components_

        return db_labels, unique_labels, num_clusters, cluster_centers


    def apply_lane_features_cluster(binary_seg_result, instance_seg_result):

        #get embedding features and coordinates
        lane_embedding_features, lane_coordinates=\
             self.get_lane_embedding_features( binary_seg_result, instance_seg_result)
        
        #apply dbscan cluster
        db_labels, unique_labels, num_clusters, cluster_centers =\
             self.embedding_features_dbscan_cluster(lane_embedding_features)
        assert db_labels is not None 

        mask= np.zeros((binary_seg_result.shape[0], binary_seg_result.shape[1], 3), np.uint8)

        true_lane_coordinates= []
        for index, label in enumerate(unique_labels.tolist()):
            if label==-1: continue
            idx= np.where(db_labels==label)
            pix_coord_idx= tuple(())




class LaneNetPostProcessor(object):

    def __init__(self, ipm_remap_file_path = 'tusimple_ipm_remap.yml'):
        assert os.path.exists(ipm_remap_file_path), "{:s} doesnot exist".format(ipm_remap_file_path)

        self.cluster= LaneCluster()
        self.ipm_remap_file_path= ipm_remap_file_path 

        remap_file= self.load_remap_matrix()         
        self.remap_to_ipm_x= remap_file['remap_to_ipm_x']
        self.remap_to_ipm_y= remap_file['remap_to_ipm_y']

        self.color_map= []

    
    def load_remap_matrix(self):
        fs= cv2.FileStorage(self.ipm_remap_file_path, cv2.FILE_STORAGE_READ)

        remap_to_ipm_x= fs.getNode('remap_ipm_x').mat() 
        remap_to_ipm_y= fs.getNode('remap_ipm_y').mat() 

        fs.release() 

        ret= {
            "remap_to_ipm_x": remap_to_ipm_x,
            "remap_to_ipm_y": remap_to_ipm_y 
        }

        return ret

    def postprocess(self, binary_seg_result, instance_seg_result, source_image):
        
        #convert binary_seg_result range from [0, 1] to [0, 255]
        binary= np.array(binary_seg_result * 255, dtype= np.uint8) 

        #apply morophology operation to fill in holes
        binary= morphological_process(binary) 

        _, labels, stats, _= connect_components_analysis(binary)
        min_area_threshold= 100
        for index, stat in enumerate(stats):
            if stat[4] <= min_area_threshold:
                idx= np.where(labels==index) 
                binary[idx]= 0 
        
        mask_image, lane_coords= self.cluster.apply_lane_features_cluster(
            binary_seg_result= binary,
            instance_seg_result= instance_seg_result
        )

        assert mask_image, "mask_image is None, ynf3 kda!"




