

import tensorflow as tf

arr= tf.constant([0, 0, 0, 1, 1, 0], shape= [1, 6])
onehot= tf.one_hot(arr, depth= 2)
logits= tf.ones_like(onehot)
arr_flattened= tf.reshape(arr, shape= [arr.shape.as_list()[0] * arr.shape.as_list()[1]])

unique_labels, unique_ids, counts= tf.unique_with_counts(arr_flattened)
inverse_weights= tf.cast(tf.divide(1.0, tf.log(tf.add(tf.divide(counts, tf.reduce_sum(counts)), 1.02))), tf.float32)
inverse_weights_sum= tf.reduce_sum(inverse_weights)

print("onehot shape:", onehot.shape.as_list())
print("inverse_weights shape:", inverse_weights.shape.as_list())
print("logits shape:", logits.shape.as_list())

loss_weights= tf.reduce_sum(tf.multiply(onehot, inverse_weights))
loss= tf.losses.softmax_cross_entropy(onehot_labels= onehot, logits= onehot, weights= loss_weights)


with tf.Session() as sess:
    
    
    onehot_output, logits_output, unique_labels_output, unique_ids_output, counts_output, inverse_weights_output, inverse_weights_sum_output, \
    loss_weights_output, loss_output= sess.run([onehot, logits, unique_labels, unique_ids, counts, inverse_weights, inverse_weights_sum, loss_weights, loss])
    
    print("onehot:", onehot_output)
    print("logits:", logits_output)
    
    print("unique-labels:", unique_labels_output)
    print("unique_ids:", unique_ids_output)
    print("counts:", counts_output)

    print("inverse_weights:", inverse_weights_output)
    print("inverse_sum_output:", inverse_weights_sum_output)

    print("loss weights output:", loss_weights_output)
    print("loss output:", loss_output)


