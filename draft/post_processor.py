
from my_postprocessor import *






def test():

    binary= read_image('results/binary1.png')
    source= read_image("results/source1.jpg")

    pp= PostProcessor()
    mask, ipm_mask= pp.post_process(binary, source)  
    
    save_image('results', 'test', mask) 
    save_image('results', 'test2', ipm_mask) 
    print("done")




















if __name__ == "__main__":
    
    test() 