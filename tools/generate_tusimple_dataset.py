
'''
generates usable dataset format from tuSimple dataset
'''

import argparse
import glob
import json
import os
import shutil
import cv2
import numpy as np
import time


def init_args():

    parser= argparse.ArgumentParser()
    parser.add_argument('--tusimple_dir', type= str, help= 'the original unzipped tuSimple dataset')

    return parser.parse_args()

def generate_trian_sample(src_dir, images_dir, binary_dir, instance_dir):
    
    #check if directories exist
    if not os.path.exists(src_dir) or not os.path.exists(images_dir) or not os.path.exists(binary_labels) or not os.path.exists(instance_labels):
        print("one or all directories paths doensnot exist")
        return

    #open a text file inside output_training folder called train.txt
    #to store path for each image and its corresponding binary and instance labels
    with open('{:s}/output_training/train.txt'.format(src_dir), 'w') as train_file:
        #for each image in images_dir
        for image_name in os.listdir(images_dir):
            #collect the image and the corresponding binary and instance labels,
            #this is easy because image and its labels share file name
            image_path= os.path.join(images_dir, image_name)
            binary_path= os.path.join(binary_dir, image_name)
            instance_path= os.path.join(instance_dir, image_name)

            #check the paths exit and end with .png
            if not os.path.exists(image_path) or not os.path.exists(binary_path) or not os.path.exists(instance_path):
                print("one or all of file paths for {:s} doesnot exist".format(image_path))
                return
            if not image_path.endswith('.png') or not binary_path.endswith('.png') or not instance_path.endswith('.png'):
                print("file extension for {:s} is corrupted".format(image_path))
                return
            
            #gather their names in a line separated with space
            entry= "{:s} {:s} {:s}".format(image_path, binary_path, instance_path)

            #write the line in the train.txt file and add '\n'
            train_file.write(entry + '\n')

def process_json_file(json_path, tusimple_dataset_path, dst_images_path, dst_binary_path, dst_instance_path):

    assert os.path.exists(json_path), '{:s} doesnot exist'.format(json_path)
    t= str(int(time.time()))


    image_nums= len(os.listdir(dst_images_path))

    with open(json_path, 'r') as json_file:
        for line_index, line in enumerate(json_file):
            json_entry = json.loads(line)

            src_image_path= os.path.join(tusimple_dataset_path, json_entry['raw_file'])
            h_samples= json_entry['h_samples']
            lanes= json_entry['lanes']

            #new image name
            new_image_name= '{:s}_{:d}.png'.format(t, line_index)
            #upload source image and create binary and instance images
            src_image= cv2.imread(src_image_path, cv2.IMREAD_COLOR)
            dst_binary_image= np.zeros([src_image.shape[0], src_image.shape[1]], np.uint8)
            dst_instance_image= np.zeros([src_image.shape[0], src_image.shape[1]], np.uint8)
            #for each lane: extract x and y points for polylines drawing
            for lane_index, lane in enumerate(lanes):
                assert len(h_samples) == len(lane), 'lane x and y points lengths are different'
                
                lane_x= []
                lane_y= []

                for index in range(len(lane)):
                    if lane[index] == -2: continue
                    lane_x.append(lane[index])
                    lane_y.append(h_samples[index])
                if not lane_x: continue

                #some formating for polylines drawing
                lane_pts= np.vstack((lane_x, lane_y)).transpose()
                lane_pts= np.array([lane_pts], np.int64)
                #draw polylines on new images
                cv2.polylines(dst_binary_image, lane_pts, isClosed= False, color=255, thickness= 5)
                cv2.polylines(dst_instance_image, lane_pts, isClosed= False, color= lane_index * 50 + 20, thickness= 5)
                
                
            #create paths and save new images
            dst_image_path= os.path.join(dst_images_path, new_image_name)
            dst_binary_image_path= os.path.join(dst_binary_path, new_image_name)
            dst_instance_image_path= os.path.join(dst_instance_path, new_image_name)
            cv2.imwrite(dst_image_path, src_image)
            cv2.imwrite(dst_binary_image_path, dst_binary_image)
            cv2.imwrite(dst_instance_image_path, dst_instance_image)



def process_tuSimple_dataset(tusimple_dir):

    # make 2 directories to save output trianing and testing images
    training_folder_path= os.path.join(tusimple_dir, 'output_training')
    testing_folder_path= os.path.join(tusimple_dir, 'output_testing')
    os.makedirs(training_folder_path, exist_ok= True)
    os.makedirs(testing_folder_path, exist_ok= True)

    #make a copy of every json file to the newly generated training folder
    for json_labels_file_path in glob.glob('{:s}/label*.json'.format(tusimple_dir)):
        json_labels_file_name= os.path.split(json_labels_file_path)[1]
        shutil.copyfile(json_labels_file_path, os.path.join(training_folder_path, json_labels_file_name))
        
    #make a copy of every json file to the newly generated testing folder
    for json_labels_file_path in glob.glob('{:s}/test*.json'.format(tusimple_dir)):
        json_labels_file_name= os.path.split(json_labels_file_path)[1]
        shutil.copyfile(json_labels_file_path, os.path.join(testing_folder_path, json_labels_file_name))
    

    #create subdirs withing newly generated training folder to contain training images
    images_dir= os.path.join(training_folder_path, 'images')
    binary_dir= os.path.join(training_folder_path, 'binary_labels')
    instance_dir= os.path.join(training_folder_path, 'instance_labels')
    os.makedirs(images_dir, exist_ok= True)
    os.makedirs(binary_dir, exist_ok= True)
    os.makedirs(instance_dir, exist_ok= True)

    for json_labels_file_path in glob.glob('{:s}/*.json'.format(training_folder_path)):
        process_json_file(json_labels_file_path, tusimple_dir, images_dir, binary_dir, instance_dir)
        print("processed {:s}".format(os.path.split(json_labels_file_path)[1]))

    generate_trian_sample(tusimple_dir, images_dir, binary_dir, instance_dir)

if __name__ == '__main__':
    
    #args= init_args()
    tusimple_dir= r"C:\Users\gad\Desktop\data\train"
    images_dir= r"C:\Users\gad\Desktop\data\train\output_training\images"
    binary_labels= r"C:\Users\gad\Desktop\data\train\output_training\binary_labels"
    instance_labels= r"C:\Users\gad\Desktop\data\train\output_training\instance_labels"

    generate_trian_sample(tusimple_dir, images_dir, binary_labels, instance_labels)
    #process_tuSimple_dataset(tusimple_dir)
    