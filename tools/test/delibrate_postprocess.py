

from test_utils import * 



def analyze_binary(binary):
    pp = FakePostProcessor() 
    pp.process(binary) 
    

def run(path) :

    binary = get_image_paths_list(path)[0] 
    print(binary)

    binary = read_image(binary)
    analyze_binary(binary) 


path = r'C:\Users\gad\Desktop\repos\VOLO\images\binary'
run(path) 