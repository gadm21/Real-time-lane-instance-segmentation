

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

def draw_beautiful_circles(image, lane_pts, start_y, colors):

    for index, single_lane_pts in enumerate(lane_pts):
        xs = np.array(single_lane_pts, dtype=np.float32)[:, 0]
        ys = np.array(single_lane_pts, dtype=np.float32)[:, 1]
        idx= np.where(ys >= start_y) 
        ys= ys[idx]
        xs= xs[idx]

        thickness= 7
        points= len(xs) 
        for i, y in enumerate(ys):
            
            new_thickness= int(thickness * (i / points))
            cv2.circle(image, (xs[i], y), new_thickness, colors[min(index+1, len(colors)-1)], -1 )

#morphological process fills the holes in the binary segmentation result
def morphological_process(image, kernel_size= 5):
    
    assert (len(image.shape) < 3), "binary image must have a single channel"
    image= np.array(image, np.uint8) 

    kernel= cv2.getStructuringElement(shape= cv2.MORPH_ELLIPSE, ksize= (kernel_size, kernel_size))
    filled_holes= cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=1 )

    return filled_holes 

#converts an image to grayscale 
def toGray(image):
    if len(image.shape) == 3:
        gray_image= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else: gray_image= image

    return gray_image

#remove small components
def connect_components_analysis(image):
    gray_image= toGray(image)
    return cv2.connectedComponentsWithStats(gray_image, connectivity= 8, ltype= cv2.CV_32S)




class LaneCluster(object):

    def __init__(self):
        self.color_map = [np.array([255, 0, 0]),
                    np.array([0, 255, 0]),
                    np.array([0, 0, 255]),
                    np.array([125, 125, 0]),
                    np.array([0, 125, 125]),
                    np.array([125, 0, 125]),
                    np.array([50, 100, 50]),
                    np.array([100, 50, 100])]
    
    #get lane embedding features according to binary
    @staticmethod
    def get_lane_embedding_features(binary, instance):
        
        #lane_embedding_features are the values in the instance_seg_image on the lanes coords
        idx= np.where(binary == 255)
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
            print("clustering error")
            return None, None, None, None
        
        db_labels= db.labels_
        unique_labels= np.unique(db_labels) 
        #num_clusters= len(unique_labels)
        #cluster_centers= db.components_

        return db_labels, unique_labels #, num_clusters, cluster_centers


    def apply_lane_features_cluster(self, binary_seg_result, instance_seg_result):

        #get embedding features and coordinates
        lane_embedding_features, lane_coordinates=\
             self.get_lane_embedding_features( binary_seg_result, instance_seg_result)
        
        #apply dbscan cluster
        db_labels, unique_labels= self.embedding_features_dbscan_cluster(lane_embedding_features)
        assert db_labels is not None 

        mask= np.zeros((binary_seg_result.shape[0], binary_seg_result.shape[1], 3), np.uint8)

        true_lane_coordinates= []
        for index, label in enumerate(unique_labels.tolist()):
            if label==-1: continue
            idx= np.where(db_labels==label)
            pix_coord_idx= tuple((lane_coordinates[idx][:, 1], lane_coordinates[idx][:, 0]))
            mask[pix_coord_idx]= self.color_map[index]
            true_lane_coordinates.append(lane_coordinates[idx])
        
        return mask, true_lane_coordinates



class LaneNetPostProcessor(object):

    def __init__(self, ipm_remap_file_path = 'tusimple_ipm_remap.yml'):
        assert os.path.exists(ipm_remap_file_path), "{:s} doesnot exist".format(ipm_remap_file_path)

        self.cluster= LaneCluster()
        self.ipm_remap_file_path= ipm_remap_file_path 

        remap_file= self.load_remap_matrix()         
        self.remap_to_ipm_x= remap_file['remap_to_ipm_x']
        self.remap_to_ipm_y= remap_file['remap_to_ipm_y']

        self.color_map = [np.array([255, 0, 0]),
                    np.array([0, 255, 0]),
                    np.array([0, 0, 255]),
                    np.array([125, 125, 0]),
                    np.array([0, 125, 125]),
                    np.array([125, 0, 125]),
                    np.array([50, 100, 50]),
                    np.array([100, 50, 100])]

    
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

    def postprocess(self, binary_seg_result, instance_seg_result, source_image, data_source= "tusimple"):
        
        intermediate_path= "images/intermediate"
        os.makedirs(intermediate_path, exist_ok=True) 

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

        assert mask_image is not None, "mask_image is None, ynf3 kda!"

        #lane curve fitting
        fit_params= []
        src_lane_pts= []
        for lane_index, coordinates in enumerate(lane_coords):
            tmp_mask = np.zeros(shape=(720, 1280), dtype=np.uint8)
            idx= tuple((np.int_(coordinates[:, 1] * 720 / 256), np.int_(coordinates[:, 0] * 1280 / 512)))
            tmp_mask[idx] = 255
            
            tmp_ipm_mask= cv2.remap(tmp_mask, self.remap_to_ipm_x, self.remap_to_ipm_y, interpolation= cv2.INTER_NEAREST)

            nonzero_y= tmp_ipm_mask.nonzero()[0]
            nonzero_x= tmp_ipm_mask.nonzero()[1]

            lane_parameters= np.polyfit(nonzero_y, nonzero_x, 2)
            fit_params.append(lane_parameters)

            [ipm_image_height, ipm_image_width]= tmp_ipm_mask.shape
            plot_y= np.linspace(10, ipm_image_height, ipm_image_height-10)
            fit_x= lane_parameters[0] * plot_y**2 + lane_parameters[1] * plot_y + lane_parameters[2]

            lane_pts = []
            for index in range(0, plot_y.shape[0], 5):
                src_x = self.remap_to_ipm_x[ int(plot_y[index]), int(np.clip(fit_x[index], 0, ipm_image_width - 1))]
                if src_x <= 0: continue

                src_y = self.remap_to_ipm_y[ int(plot_y[index]), int(np.clip(fit_x[index], 0, ipm_image_width - 1))]
                src_y = src_y if src_y > 0 else 0

                lane_pts.append([src_x, src_y])

            src_lane_pts.append(lane_pts)
            
        # tusimple test data sample point along y axis every 10 pixels
        source_image_width = source_image.shape[1]
        
        start_plot_y= 300
        colors= [self.color_map[index].tolist() for index in range(len(src_lane_pts))]
        draw_beautiful_circles(source_image, src_lane_pts, start_plot_y, colors)
            
        '''

        step = int(math.floor((end_plot_y - start_plot_y) / 1))
        for plot_y in np.linspace(start_plot_y, end_plot_y, step):
            diff = single_lane_pt_y - plot_y
            fake_diff_bigger_than_zero = diff.copy()
            fake_diff_smaller_than_zero = diff.copy()
            fake_diff_bigger_than_zero[np.where(diff <= 0)] = float('inf')
            fake_diff_smaller_than_zero[np.where(diff > 0)] = float('-inf')
            idx_low = np.argmax(fake_diff_smaller_than_zero)
            idx_high = np.argmin(fake_diff_bigger_than_zero)

            previous_src_pt_x = single_lane_pt_x[idx_low]
            previous_src_pt_y = single_lane_pt_y[idx_low]
            last_src_pt_x = single_lane_pt_x[idx_high]
            last_src_pt_y = single_lane_pt_y[idx_high]

            if previous_src_pt_y < start_plot_y or last_src_pt_y < start_plot_y or \
                    fake_diff_smaller_than_zero[idx_low] == float('-inf') or \
                    fake_diff_bigger_than_zero[idx_high] == float('inf'):
                continue

            interpolation_src_pt_x = (abs(previous_src_pt_y - plot_y) * previous_src_pt_x +
                                        abs(last_src_pt_y - plot_y) * last_src_pt_x) / \
                                        (abs(previous_src_pt_y - plot_y) + abs(last_src_pt_y - plot_y))
            interpolation_src_pt_y = (abs(previous_src_pt_y - plot_y) * previous_src_pt_y +
                                        abs(last_src_pt_y - plot_y) * last_src_pt_y) / \
                                        (abs(previous_src_pt_y - plot_y) + abs(last_src_pt_y - plot_y))

            if interpolation_src_pt_x > source_image_width or interpolation_src_pt_x < 10:
                continue

            lane_color = self.color_map[index].tolist()
            cv2.circle(source_image, (int(interpolation_src_pt_x),
                                        int(interpolation_src_pt_y)), 5, lane_color, -1)
            '''
        
        ret = {
            'mask_image': mask_image,
            'fit_params': fit_params,
            'source_image': source_image,
        }

        return ret


class LaneNetAdvancedPostProcessor(object):

    def __init__(self, ipm_remap_file_path= 'tusimple_ipm_remap.yml'):

        assert os.path.exists(ipm_remap_file_path), "{:s} doesnot exist".format(ipm_remap_file_path)
        remap_file= self.load_remap_matrix(ipm_remap_file_path) 

    def load_remap_matrix(self, ipm_remap_file_path):
        fs= cv2.FileStorage(ipm_remap_file_path, cv2.FILE_STORAGE_READ)

        self.remap_to_ipm_x= fs.getNode('remap_ipm_x').mat() 
        self.remap_to_ipm_y= fs.getNode('remap_ipm_y').mat() 

        gs.release() 

    def smart_cluster(binary):

        def get_hist(image):
            return np.sum(image[image.shape[0]//2:, :], axis=0)
    
        

    def postprocess(self, binary, source):
        intermediate_path= 'images/intermediate'
        os.makedirs(intermediate_path, exist_ok= True) 

        binary= np.array(binary* 255, dtype= np.uint8) 

        binary= morphological_process(binary) 

        _, stats, labels, _= connect_components_analysis(binary)
        min_area_threshold= 100
        for index, stat in enumerate(stats):
            if stat[4] < min_area_threshold: 
                binary[labels==index] = 0

        lanes_coordinates= self.smart_cluster(binary)


