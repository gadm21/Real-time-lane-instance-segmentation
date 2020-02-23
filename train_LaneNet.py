


def train_LaneNet (dataset_dir, weights_path):

    ''' waiting
    train_dataset= LaneNet_data_feed_pipeline.LaneNetDataFeeder(dataset_dir, flags= 'train')
    val_dataset= LaneNet_data_feed_pipeline.LaneNetDataFeeder(dataset_dir, flags= 'val')
    '''


    with tf.device('/gpu:1'):
        