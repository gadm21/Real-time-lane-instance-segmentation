import numpy as np 
import tensorflow as tf

c = np.array([[1,2,3,4], [5,6,7,8], [4,3,2,1]])
print(c.shape)
new_c= np.reshape(c,  (-1, 1))

print(new_c.shape)