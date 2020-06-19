import numpy as np 

v = np.array([1,-2,0.5]) 

s1 = np.exp(v[0]) / np.sum(np.exp(v)) 
s2 = np.exp(v[1]) / np.sum(np.exp(v)) 
s3 = np.exp(v[2]) / np.sum(np.exp(v)) 

print(s1)
print(s2) 
print(s3) 