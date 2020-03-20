
import os 
import sys 
sys.path.append(os.getcwd())

import numpy as np 
import cv2 
import pickle 

from LaneNet.LaneNet_model.LaneNet_PostProcessor import LaneNetPostProcessor

def load_images():
    
    with open("images/binary.pickle", "rb") as binary:
        binary_image= pickle.load(binary)
    
    with open("images/instance.pickle", "rb") as instance:
        instance_image= pickle.load(instance) 
        
    source_image= cv2.imread("images/source_image.png")

    return source_image, binary_image, instance_image 


def test_postprocessor():

    source, binary, instance= load_images()
    postprocessor= LaneNetPostProcessor()
    ret= postprocessor.postprocess(binary, instance, source) 

    cv2.imshow("R", ret) 
    cv2.imshow("J", np.array(binary * 255, dtype= np.uint8)) 
    cv2.waitKey(0) 
    cv2.destroyAllWindows() 
    




if __name__ == "__main__":
    test_postprocessor()
    print("done") 