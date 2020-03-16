import numpy as np
import tensorflow as tf    

# 1000 sequence in the length of 100
matrix = tf.placeholder(tf.int32, shape=(100, 1000), name="input_matrix")
matrix_rows = tf.shape(matrix)[0]
print("rows:", matrix_rows.shape.as_list())
ta = tf.TensorArray(dtype=tf.int32, size=matrix_rows)

init_state = (0, ta)
condition = lambda i, _: i < matrix_rows
body = lambda i, ta: (i + 1, ta.write(i, matrix[i] * 2))
n, ta_final = tf.while_loop(condition, body, init_state)
# get the final result
ta_final_result = ta_final.stack()

# run the graph
with tf.Session() as sess:
    # print the output of ta_final_result
    arr= sess.run(ta_final_result, feed_dict={matrix: np.ones(shape=(100,1000), dtype=np.int32)}) 
    print(arr.shape)