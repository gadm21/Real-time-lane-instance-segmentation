
import os
import sys
#sys.path.append(os.getcwd())


import glob
import random
import argparse
from global_config import cfg

import tensorflow as tf
from . import tf_io_pipeline_tools

CFG= cfg
'''
firstly, LaneNetDataProducer generates tfrecords from dataset, then LaneNetDataFeeder, reads
the tfrecords files and generate data to be trained on to the LaneNet model
'''


def init_args():

    parser= argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, help= 'the source nsfw data dir path')
    parser.add_argument('--tfrecords_dir', type= str, help= 'the dir path to save converted tfrecords')

    return parser.parse_args() 

#convert raw image files into tfrecords
class LaneNetDataProducer(object):
     
    def __init__(self, dataset_dir):
        self.dataset_dir= dataset_dir

        #directory of training data and their labels
        self.gt_image_dir= os.path.join(dataset_dir, 'gt_image')
        self.gt_binary_image_dir= os.path.join(dataset_dir, 'gt_binary_image')
        self.gt_instance_image_dir= os.path.join(dataset_dir, 'gt_instance_image')

        #files that carry paths of each image and its corresponding label images
        self.train_example_index_file_path= os.path.join(dataset_dir, 'train.txt')
        self.test_example_index_file_path= os.path.join(dataset_dir, 'test.txt')
        self.val_example_index_file_path= os.path.join(dataset_dir, 'val.txt')
    

    #generate tfrecords file contents
    def generate_tfrecords(self, save_dir, step_size= 10000): # step_size is the batch size

        #this inner function is applied to 'train.txt', 'val.txt', and 'test.txt'
        #to extract the information in them
        def read_training_example_index_file(index_file_path):
            example_gt_path_info= []
            example_gt_binary_path_info= []
            example_gt_instance_path_info= []

            with open(index_file_path, 'r') as file:
                for line in file:
                    example_info= line.rstrip('\r').rstrip('\n').split(' ')
                    example_gt_path_info.append(example_info[0])
                    example_gt_binary_path_info.append(example_info[1])
                    example_gt_instance_path_info.append(example_info[2])
            
            ret= {
                'gt_path_info': example_gt_path_info,
                'gt_binary_path_info': example_gt_binary_path_info,
                'gt_instance_path_info': example_gt_instance_path_info 
            }

            return rt

        def create_batchSized_tfrecords(example_gt_paths, example_gt_binary_paths, example_gt_instance_paths, flags= 'train'):
            batchSized_example_gt_paths= []
            batchSized_example_gt_binary_paths= []
            batchSized_example_gt_instance_paths= []
            batchSized_tfrecords_paths= []

            for i in range(0, len(example_gt_paths), step_size):
                #even if (i+step_size) > len(example_gt_paths), the operator [num:num]
                # does a boundary check before indexing
                batchSized_example_gt_paths.append(example_gt_paths[i: i+step_size])
                batchSized_example_gt_binary_paths.append(example_gt_binary_paths[i: i+step_size])
                batchSized_example_gt_instance_paths.append(example_gt_instance_paths[i: i+step_size])

                if (i+step_size) > len(example_gt_paths):
                    batchSized_tfrecords_paths.append(
                        os.path.join(save_dir, '{:s}_{:d}_{:d}.tfrecords'.format(flags, i, len(example_gt_paths))))
                else:
                    batchSized_tfrecords_paths.append(
                        os.path.join(save_dir, '{:s}_{:d}_{:d}.tfrecords'.format(flags, i, i+step_size)))
            
            ret= {
                'gt_paths': batchSized_example_gt_paths,
                'gt_binary_paths': batchSized_example_gt_binary_paths,
                'gt_instance_paths': batchSized_example_gt_instance_paths,
                'tfrecords_paths': batchSized_tfrecords_paths
            }

            return ret

        os.makedirs(save_dir, exist_ok=True) #exist_ok flag doesn't raise an error if dir exists

        #collecting train images paths info (images and their corresponding labels)
        train_image_paths_info= read_training_example_index_file(self.train_example_index_file_path)
        #split training images according to step_size (batch size)
        train_split_result= create_batchSized_tfrecords(train_image_paths_info['gt_path_info'],
                                                        train_image_paths_info['gt_binary_path_info'],
                                                        train_image_paths_info['gt_instance_path_info'])
        batchSized_gt_paths= train_split_result['gt_paths']
        batchSized_gt_binary_paths= train_split_result['gt_binary_paths']
        batchSized_gt_instance_paths= train_split_result['gt_instance_paths']
        tfrecord_paths= train_split_result['tfrecords_paths']
        for index, batchSized_gt_path in enumerate(batchSized_gt_paths):
            tf_io_pipeline_tools.write_example_tfrecords(batchSized_gt_path, 
                                                        batchSized_gt_binary_paths[index],
                                                        batchSized_gt_instance_paths[index],
                                                        tfrecord_paths[index])
        

        val_image_paths_info= read_training_example_index_file(self.val_example_index_file_path)
        val_split_result= create_batchSized_tfrecords(val_image_paths_info['gt_paths_info'],
                                                        val_image_paths_info['gt_binary_paths_info'],
                                                        val_image_paths_info['gt_instance_paths_info'])
        batchSized_gt_paths= val_split_result['gt_paths']
        batchSized_gt_binary_paths= val_split_result['gt_binary_paths']
        batchSized_gt_instance_paths= val_split_result['gt_instance_paths']
        tfrecord_paths= val_split_result['tfrecords_paths']
        for index, batchSized_gt_path in enumerate(batchSized_gt_paths):
            tf_io_pipeline_tools.write_example_tfrecords(batchSized_gt_path,
                                                        batchSized_gt_binary_paths[index],
                                                        batchSized_gt_instance_paths[index],
                                                        tfrecord_paths[index])

        test_image_paths_info= read_training_example_index_file(self.test_example_index_file_path)
        test_split_result= create_batchSized_tfrecords(test_image_paths_info['gt_paths_info'],
                                                        test_image_paths_info['gt_binary_paths_info'],
                                                        test_image_paths_info['gt_instance_paths_info'])
        batchSized_gt_paths= test_split_result['gt_paths']
        batchSized_gt_binary_paths= test_split_result['gt_binary_paths']
        batchSized_gt_instance_paths= test_split_result['gt_instance_paths']
        tfrecord_paths= test_split_result['tfrecords_paths']
        for index, batchSized_gt_path in enumerate(batchSized_gt_paths):
            tf_io_pipeline_tools.write_example_tfrecords(batchSized_gt_path,
                                                        batchSized_gt_binary_paths[index],
                                                        batchSized_gt_instance_paths[index],
                                                        tfrecord_paths[index])        
        return    

    def is_source_data_complete(self):
        return \
            os.path.exists(self.gt_image_dir) and \
            os.path.exists(self.gt_binary_image_dir) and \
            os.path.exists(self.gt_instance_image_dir)


    def _is_training_sample_index_file_complete(self):
        """
        Check if the training sample index file is complete
        :return:
        """
        return \
            ops.exists(self.train_example_index_file_path) and \
            ops.exists(self.test_example_index_file_path) and \
            ops.exists(self.val_example_index_file_path)

    def _generate_training_example_index_file(self):
        """
        Generate training example index file, split source file into 0.85, 0.1, 0.05 for training,
        testing and validation. Each image folder are processed separately
        :return:
        """

        def _gather_example_info():
            """

            :return:
            """
            _info = []

            for _gt_image_path in glob.glob('{:s}/*.png'.format(self.gt_image_dir)):
                _gt_binary_image_name = ops.split(_gt_image_path)[1]
                _gt_binary_image_path = ops.join(self.gt_binary_image_dir, _gt_binary_image_name)
                _gt_instance_image_name = ops.split(_gt_image_path)[1]
                _gt_instance_image_path = ops.join(self.gt_instance_image_dir, _gt_instance_image_name)

                assert ops.exists(_gt_binary_image_path), '{:s} not exist'.format(_gt_binary_image_path)
                assert ops.exists(_gt_instance_image_path), '{:s} not exist'.format(_gt_instance_image_path)

                _info.append('{:s} {:s} {:s}\n'.format(
                    _gt_image_path,
                    _gt_binary_image_path,
                    _gt_instance_image_path)
                )

            return _info

        def _split_training_examples(_example_info):
            random.shuffle(_example_info)

            _example_nums = len(_example_info)

            _train_example_info = _example_info[:int(_example_nums * 0.85)]
            _val_example_info = _example_info[int(_example_nums * 0.85):int(_example_nums * 0.9)]
            _test_example_info = _example_info[int(_example_nums * 0.9):]

            return _train_example_info, _test_example_info, _val_example_info

        train_example_info, test_example_info, val_example_info = _split_training_examples(_gather_example_info())

        random.shuffle(train_example_info)
        random.shuffle(test_example_info)
        random.shuffle(val_example_info)

        with open(ops.join(self.dataset_dir, 'train.txt'), 'w') as file:
            file.write(''.join(train_example_info))

        with open(ops.join(self.dataset_dir, 'test.txt'), 'w') as file:
            file.write(''.join(test_example_info))

        with open(ops.join(self.dataset_dir, 'val.txt'), 'w') as file:
            file.write(''.join(val_example_info))

        #log.info('Generating training example index file complete')

        return

#reads training examples from tfrecords for (nsfw?) model
class LaneNetDataFeeder(object):

    def __init__(self, dataset_dir, flags):
        self.dataset_dir= dataset_dir.lower()
        self.dataset_flags= flags.lower()
        self.tfrecords_dir= os.path.join(dataset_dir, 'tf_records')
    

    def inputs(self, batch_size, num_epochs):
        
        dir= r"C:\Users\gad\Downloads\Compressed\lanenet-lane-detection\data\data_records"
        tfrecords_file_paths= glob.glob('{:s}\{:s}*.tfrecords'.format(dir, self.dataset_flags))
        random.shuffle(tfrecords_file_paths)
        

        with tf.name_scope('input_tensor'): #meaning?

            dataset= tf.data.TFRecordDataset(tfrecords_file_paths)

            dataset= dataset.map(map_func= tf_io_pipeline_tools.decode,num_parallel_calls= CFG.TRAIN.CPU_MULTI_PROCESS_NUMS)
            if self.dataset_flags == 'test':
                dataset= dataset.map(map_func= tf_io_pipeline.augment_for_test,num_parallel_calls=CFG.TRAIN.CPU_MULTI_PROCESS_NUMS)
            else:
                dataset= dataset.map(map_func= tf_io_pipeline_tools.augment_for_train, num_parallel_calls=CFG.TRAIN.CPU_MULTI_PROCESS_NUMS)
            
            dataset= dataset.map(map_func= tf_io_pipeline_tools.normalize, num_parallel_calls=CFG.TRAIN.CPU_MULTI_PROCESS_NUMS)

            if self.dataset_flags != 'test':
                dataset= dataset.shuffle(buffer_size= 1000)
                dataset= dataset.repeat()
            
            dataset= dataset.batch(batch_size, drop_remainder= True)

            iterator= dataset.make_one_shot_iterator()

        return iterator.get_next(name= '{:s}_IteratorGetNext'.format(self.dataset_flags))





if __name__ == "__main__":
    args= init_args()

    assert os.path.exists(args.dataset_dir), '{:s} doesnot exist'.format(args.dataset_dir)

    producer= LaneNetDataProducer(dataset_dir= args.dataset_dir)
    producer.generate_tfrecords(save_dir= args.tfrecords_dir, step_size= 1000)

