


def train_LaneNet (dataset_dir, weights_path):

    ''' waiting
    first guess:: this function takes the dataset directory and returns clean chunks, 
    ready to be used in the network (for training or validation) depending on the 
    batch size and epochs_num that is passed later to LaneNetDataFeeder.input

    train_dataset= LaneNet_data_feed_pipeline.LaneNetDataFeeder(dataset_dir, flags= 'train')
    val_dataset= LaneNet_data_feed_pipeline.LaneNetDataFeeder(dataset_dir, flags= 'val')
    '''


    train_net= LaneNet.LaneNet(phase= 'train', reuse= False)
    val_net= LaneNet.LaneNet(phase= 'val', reuse= True)


