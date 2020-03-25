
from helper import *






def test():

    image= read_image('images/binary.png')
    
    pp= PostProcessor()
    new_image= pp.post_process(image) 
    
    show_image(new_image) 
    save_image('results', 'final', new_image) 

























if __name__ == "__main__":
    
    test() 