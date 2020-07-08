
from test_utils import * 
import argparse






def init_args() : 
    parser = argparse.ArgumentParser() 
    parser.add_argument('--image') 
    return parser.parse_args() 






def predict(image, save = None, display = True, weights = None):

    if not weights : weights = weights_path 
    image = np.array([normalize( resize_image(image, (512,256)))])  

    input = tf.placeholder(dtype = tf.float32, shape = [1, 256,512,3]) 
    net = LaneNet()
    binary, score = net.inference(input)

    with tf.Session() as sess : 
        load_weights(sess, weights) 
        binary_out, score_out = sess.run([binary, score], {input: image}) 
        binary_image, score_image = binary_out[0], score_out[0]

    binary_image = process_binary(denormalize(binary_image) )

    if save : save_image(save, 'binary', binary_image) 
    if display : show_image(binary_image) 

    return binary_image


if __name__ == "__main__" : 

    args = init_args() 
    image = read_image(args.image) 

    binary = predict(image, save=None, display= False) 
    pp = PostProcessor() 
    ret = pp.process(binary) 

    show_image(ret['mask'])         