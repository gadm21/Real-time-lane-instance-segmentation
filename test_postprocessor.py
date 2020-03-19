

import numpy as np 
import cv2 
import pickle 

def load_images():
    
    with open("images/binary.pickle", "rb") as binary:
        binary_image= pickle.load(binary)
    
    with open("images/instance.pickle", "rb") as instance:
        instance_image= pickle.load(instance) 
        
    source_image= cv2.imread("images/source_image.png")

    return source_image, binary_image, instance_image 


def test_postprocessor():

    source, binary, instance= load_images()

    print("source shape:", source.shape)
    print("binary shape:", binary.shape) 
    print("instance shape:", instance.shape) 






if __name__ == "__main__":
    test_postprocessor()