
import os 
import sys 
sys.path.append(os.getcwd())
import numpy as np 

from LaneNet_model.my_postprocessor import *


def colorize_lanes(binary, draw_mean = True):

    p = PostProcessor()
    lanes = p.process(binary) 
    new_image = np.zeros_like(binary) 

    color_map= [[0, 255, 0], [0, 0, 255], [255, 0, 0]]

    for i, lane in enumerate(lanes):
        c = color_map[i % len(color_map)]
        new_image = lane.colorize(binary, c, draw_mean) 
    
    return new_image 


image = read_image("images/results/binary.png")

save_image("images/results", "lanes_means", colorize_lanes(image, draw_mean = True))
save_image("images/results", "lanes_all", colorize_lanes(image, draw_mean = False))
