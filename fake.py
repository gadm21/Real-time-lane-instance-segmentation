




class Lane(object):

    def __init__(self, id=0):
        self.color_map= [[0, 0, 255],
                        [0, 255, 0],
                        [255, 0, 0],
                        [255, 255, 0]]
        self.clusters = []
        self.means= [] 
        self.window_h= 5 
        self.id= id 
        self.valid= False 
        self.image_h= None 
        self.image_w= None 
        self.remap_to_x= None 
        self.remap_to_y= None 
        self.lane_curve = None 
        self.birdeye_params = None 
    
    def get_coords(self):
        coords = [] 
        for cluster in self.clusters:
            coords_y, coords_x = cluster
            for y, x in zip(coords_y, coords_x) :
                coords.append((x,y)) 
        return np.array(coords) 

    def mean(self):
        current_mean= (int(np.mean(self.clusters[-1][0])), int(np.mean(self.clusters[-1][1])))
        prev_mean= self.means[-1]

        mean_change= (current_mean[0] - prev_mean[0], current_mean[1] - prev_mean[1]) 
        predicted_mean= (current_mean[0] + mean_change[0] , current_mean[1] + mean_change[1])
        self.means.append(predicted_mean) 

        return predicted_mean    

    def advanced_mean(self):
        current_mean= (np.mean(self.clusters[-1][0]), np.mean(self.clusters[-1][1]))
        prev_mean= self.means[-1]
        mean_change= (current_mean[0] - prev_mean[0], current_mean[1] - prev_mean[1]) 

        new_mean = (current_mean[0] + mean_change[0], int(0.2*prev_mean[1] + 0.8*current_mean[1]))
        self.means.append(new_mean) 
        return new_mean 

    def print_info(self):
        print("lane {:d} info:".format(self.id))
        print("number of pixels on this lane == {:d}".format(self.num_points()))
        print("average cluster width == {:f}".format(self.cluster_width()))
        
        print("mumber of clusters == {:d}".format(len(self.clusters)))
        print("____________________________________________")
        print() 

    def num_points(self):
        total= 0
        for cluster in self.clusters:
            total+= cluster[0].shape[0]
        return int(total)  
    
    def draw_mask(self, shape= None, color_means= False):
        if shape is None:
            shape= ( self.image_h, self.image_w,3 ) 

        mask= np.zeros(shape= shape, dtype= np.uint8)
        mask= self.colorize(mask, self.color_map[self.id%len(self.color_map)], color_means= color_means)
        #mask= resize_image(mask, shape) 
        return mask 

    def colorize(self, image, color, color_means= True):

        if color_means:
            for cluster in self.clusters:

                mean_x = np.mean(cluster[1], dtype=np.int32) 
                mean_y = np.mean(cluster[0], dtype=np.int32) 
                
                cv2.circle(image, (mean_x, mean_y), 1, color, 2)

        else:
            for cluster in self.clusters:
                image[cluster]= color 
            

        return image 

    def cluster_width(self):
        total= 0
        for cluster in self.clusters:
            low_x = np.min(cluster[1])
            high_x= np.max(cluster[1])       
            diff= high_x - low_x
            total+= diff 
        
        last_width = int(np.max(self.clusters[-1][1]) - np.min(self.clusters[-1][1]))
        average_width = int(total // len(self.clusters) )
        weighted_width = int(0.2*average_width + 0.8 * last_width)
        return weighted_width 

    def blacken(self, image) :
        for cluster in self.clusters:
            image[cluster] = 0
    
        return image 

    def get_start_point(self):
        last_mean = self.means[-1] 
        return last_mean[0]

    def complete(self, write, cluster_coords, image):

        self.clusters.append(cluster_coords) 
        self.means.append((int(np.mean(self.clusters[-1][0])), int(np.mean(self.clusters[-1][1]))))

        self.image_h, self.image_w = image.shape[0], image.shape[1]
        lanes_coords= np.where(image == 255) 
        
        lowest_lane_coord= np.min(lanes_coords[0])
        highest_lane_coord= np.max(cluster_coords[0]) 
        window_center= self.means[-1][1]
        slidingwindow = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2BGR)
        
        for window in range(highest_lane_coord, lowest_lane_coord, - self.window_h):
            margin= int(self.cluster_width())
            slidingwindow = cv2.rectangle(slidingwindow, (window_center-margin ,window-self.window_h), (window_center+margin ,window), color = (0,255,0), thickness = 1)

            window_pix= (lanes_coords[0] >= window - self.window_h) & \
                        (lanes_coords[0] < window) & \
                        (lanes_coords[1] > (window_center - margin)) & \
                        (lanes_coords[1] < (window_center + margin))
            lane_coords_within_window = (lanes_coords[0][window_pix], lanes_coords[1][window_pix])
            if lane_coords_within_window[0].shape[0] == 0 : continue  

            self.clusters.append(lane_coords_within_window) 
            window_center= self.mean()[1]

        if write: save_image('temp', 'slidingwindow', slidingwindow) 
        image= self.blacken(image) 
        #image= remove_noise(image) 
        return image 
             
    def load_remap_matrix(self, remap_file_path):

        assert os.path.exists(remap_file_path), "remap file doesnot exist"

        fs= cv2.FileStorage(remap_file_path, cv2.FILE_STORAGE_READ)
        self.remap_to_x= fs.getNode('remap_ipm_x').mat() 
        self.remap_to_y= fs.getNode('remap_ipm_y').mat()
        fs.release() 
    
    def get_curve(self):
        assert self.valid, 'lane:{:d} is not valid'.format(self.id) 

        ys = []
        xs = []
        for m in self.means:
            ys.append(m[0])
            xs.append(m[1])
        '''
        for cluster in self.clusters:
            for point in cluster:
                ys.append(point[0])
                xs.append(point[1])
        '''
                
        self.lane_curve = np.polyfit(ys, xs, 2)

        #self.lane_curve = np.polyfit(self.means[0], self.means[1], 2) 
        return self.lane_curve 

    def fit(self, remap_file_path = None):
        mask= self.draw_mask(color_means= False) 
        
        self.load_remap_matrix(remap_file_path)
        tmp_mask = resize_image(mask, (720, 1280))
        ipm_mask= cv2.remap(tmp_mask, self.remap_to_x, self.remap_to_y, interpolation= cv2.INTER_NEAREST)
       
        nonzero_y = np.array(ipm_mask.nonzero()[0]) 
        nonzero_x = np.array(ipm_mask.nonzero()[1]) 
        params = 0 #np.polyfit(nonzero_y, nonzero_x, 2) 
        
        return mask, ipm_mask, params
  




class FakePostProcessor(object):

    def __init__(self, ipm_remap_file_path='files/tusimple_ipm_remap.yml'):
        
        self.ipm_remap_file_path = ipm_remap_file_path 

        self.stride_h= -5
        self.lane_id= 0
        
        self.color_map= [[0, 0, 255],
                        [0, 255, 0],
                        [255, 0, 0],
                        [255, 255, 0]]


        self.dbscan_eps= 8
        self.dbscan_min_samples= 30
        self.db= DBSCAN(self.dbscan_eps, self.dbscan_min_samples) 
        self.lane_acceptance_factor= 0.4
    
    def give_id(self):
        self.lane_id+= 1
        return self.lane_id 

    def pre_processing(self, image):
        image= to_gray(image) 
        #image= remove_noise(image) 
        #image= morphological_process(image) 
        return image 

    def inspect_lanes(self, lanes):
        total_points= 0
        for lane in lanes:
            total_points+= lane.num_points()
        average_lane_points= total_points / len(lanes)
        min_lane_points= average_lane_points * self.lane_acceptance_factor

        for lane in lanes:
            if lane.num_points() > min_lane_points:
                lane.valid= True 
        
    def apply_clustering_on_stride(self, coords):

        ret= self.db.fit(np.array(coords).transpose())
        labels= ret.labels_
        unique_labels= np.unique(labels) 
        return labels, unique_labels 

    def process(self, binary ) :
        path = 'temp'
        if int(np.max(binary)) != 255 : binary = np.array(binary*255, dtype = np.uint8)
        else : binary = np.array(binary, dtype = np.uint8) 

        image= self.pre_processing(binary) 
        #image = resize_image(binary, (1280, 720) )
        #image_h, image_w = image.shape
        slidingwindow1_image = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2BGR)
        cluster_image = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2BGR)
        write_lost = True
        write_2 = True 
        write_cluster = True 
        lost_lane = image.copy() 

        lanes_coords= np.where(image == 255) 
        assert len(lanes_coords[0]), 'no lanes to process' 

        lowest_lane_coord= np.min(lanes_coords[0])
        highest_lane_coord= np.max(lanes_coords[0]) 
        
        lanes= []
        lanes_params = [] 

        for stride in range(highest_lane_coord, lowest_lane_coord, self.stride_h):
            lanes_coords= np.where(image == 255)
            target_within_stride= (lanes_coords[0] < stride) & (lanes_coords[0] >= (stride + self.stride_h))
            stride_lanes_coords= (lanes_coords[0][target_within_stride], lanes_coords[1][target_within_stride])
            
            if stride_lanes_coords[0].shape[0] == 0 : continue 

            labels, unique_labels= self.apply_clustering_on_stride(stride_lanes_coords) 
            for label in unique_labels:
                if label==-1:  continue             
                cluster= (labels == label)
                cluster_coords= (stride_lanes_coords[0][cluster], stride_lanes_coords[1][cluster])
                
                cluster_image[cluster_coords] = random.choice(self.color_map) 

                lane= Lane(self.give_id()) 
                image= lane.complete(write_2, cluster_coords, image) 
                write_2 = False 
                if write_lost : 
                    save_image(path, 'lostlane', image) 
                    write_lost = False
                lanes.append(lane) 

            slidingwindow1_image = cv2.line(slidingwindow1_image, (0,stride),(image.shape[1]-1,stride), color = (255,0,0), thickness = 1) 

        save_image(path,'clusters', cluster_image) 

        save_image(path, 'bigslidingwindow', slidingwindow1_image)

        self.inspect_lanes(lanes) 

        perfect_mask = np.zeros((720, 1280, 3), dtype = np.uint8) 
        mask = np.zeros((720, 1280, 3), dtype = np.uint8) 
        
        for lane in lanes :
            if not lane.valid : continue 
            
            color = self.color_map[self.give_id()%len(self.color_map)]

            coords = lane.get_coords() 
            coords_y = np.int_(coords[:,1]) 
            coords_x = np.int_(coords[:,0])
            start_point = np.min(coords_y) 
            end_point = np.max(coords_y) 
            
            params = np.polyfit(coords_y, coords_x, 2) 
            lanes_params.append(params) 
            
            poly_coords_y = np.int_(np.linspace(start_point, end_point , end_point - start_point)) 
            poly_coords_x = np.int_(np.clip(params[0]*poly_coords_y**2 + params[1]*poly_coords_y + params[2], 0, 1280-1) )
            
            mask[(coords_y, coords_x)] = color 

            lane_pts = np.vstack((poly_coords_x, poly_coords_y)).transpose() 
            lane_pts = np.array([lane_pts], np.int64) 
            cv2.polylines(perfect_mask, lane_pts, isClosed = False, color = color, thickness = 5) 
            #perfect_mask[(poly_coords_y, poly_coords_x)] = color 
        
        ret = {
            'mask': mask,
            'perfect_mask':perfect_mask,
            'lanes_params': lanes_params,
        }

        return ret 
        

