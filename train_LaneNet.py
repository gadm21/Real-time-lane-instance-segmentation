


def train_LaneNet (dataset_dir, weights_path):

 
    train_dataset= LaneNet_data_feed_pipeline.LaneNetDataFeeder(dataset_dir, flags= 'train')
    val_dataset= LaneNet_data_feed_pipeline.LaneNetDataFeeder(dataset_dir, flags= 'val')
    


    train_net= LaneNet.LaneNet(phase= 'train', reuse= False)
    val_net= LaneNet.LaneNet(phase= 'val', reuse= True)


