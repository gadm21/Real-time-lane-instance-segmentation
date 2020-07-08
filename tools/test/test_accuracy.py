

from test_utils import * 



source_path = r'C:\Users\gad\Desktop\data\train\output_training\images' 
b_label_path = r'C:\Users\gad\Desktop\data\train\output_training\binary_labels'




def precision(binary, label): 
    accuracy = np.count_nonzero(label[np.where(binary==255)])
    gt = np.count_nonzero(label) 
    precision = accuracy / gt 
    return precision 


def false_positive(binary, label):
    idx = np.where(binary==255) 
    prediction_count = len(idx[0]) 
    wrong_prediction = prediction_count - np.count_nonzero(label[idx]) 
    return wrong_prediction / prediction_count 

def false_negative(binary, label):
    idx = np.where(label==255)
    gt_count = len(idx[0])
    missed_labels = gt_count - np.count_nonzero(binary[idx]) 
    return missed_labels / gt_count 

def run(start =30, length= 101) :

    #get source & labels paths
    sources = get_image_paths_list(source_path)[start:start+length]
    b_labels= get_image_paths_list(b_label_path)[start:start+length]


    input = tf.placeholder(dtype = tf.float32, shape= [None, 256,512,3]) 
    net = LaneNet() 
    pp = PostProcessor() 
    binary, score = net.inference(input) 
    accs = [] 
    fps = [] 
    fns = [] 

    with tf.Session() as sess : 
        load_weights(sess, weights_path) 

        for i in range(len(sources)) :
            source = np.array([normalize(resize_image(read_image(sources[i]), (512,256)))])
            b_label = cv2.cvtColor(read_image(b_labels[i]), cv2.COLOR_BGR2GRAY) 

            binary_out, score_out = sess.run([binary,score], {input:source})
            binary_out, score_out = binary_out[0], score_out[0] 
            binary_out = denormalize(binary_out)
            ret = pp.process(binary_out) 
            
            acc = precision(binary_out, b_label) 
            accs.append(acc) 

            fp = false_positive(binary_out, b_label) 
            fps.append(fp) 

            fn = false_negative(binary_out, b_label) 
            fns.append(fn) 

            if i%10==0 :
                print("precision:{} false positive:{} false negative:{}".format(np.mean(accs), np.mean(fps), np.mean(fns)))



run() 

'''
a = np.array([1,2,3,4])
b = np.array([2,4,4,8]) 

idx = np.where( np.logical_and( a%2==0 , b%2==0 ) ) 

print(idx[0])
print(len(idx[0])) 
'''