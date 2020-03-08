
#Discriminative loss for instance segmentation

import tensorflow as tf
import numpy as np
import cv2

def show_me(label, prediction):
    images= [label, prediction]
    for image in images:
        print(image.shape)
        print(type(image))
        print()
        '''
        cv2.imshow('r', image)
        cv2.waitKey(0)
        cv2.destroyWindow('r')
        '''

'''
The 2nd argument to cond() & body() is prdiction not batch, but it called and treated as batch inside the function!! 
How is body using prediction[i]? 
Is prediction related to batch ?
May be prediction is array of predictions and thus the batch size is the number of predictions we should consider


why is body() using prediction[i] & correct_label[i] which are variables
in the outer function (discriminative_loss()) and not using batch & label
which are passed to it?

'''

def discriminative_loss_single(
    prediction,      #inference of network
    label,           #instance label
    feature_dim,     #feature dimension of prediction
    label_shape,     #shape of label
    delta_v,         #cut off variance distance
    delta_d,         #cut off cluster distance
    param_var,       #weight for intra cluster variance
    param_dist,      #weight for inter cluster distance
    param_reg):      #weight regularization



    label= tf.reshape(label, [label_shape[1] * label_shape[0]])
    prediction= tf.reshape(prediction, [label_shape[1] * label_shape[0]], feature_dim)

    #calculate instance nums

    unique_labels, unique_id, counts= tf.unique_with_counts(label)
    counts= tf.cast(counts, tf.float32)
    num_instances= tf.size(unique_labels)


    #calculate instance pixel embedding mean vec
    segmented_sum= tf.unsorted_segment_sum(prediction, unique_id, num_instances)
    mu= tf.div(segmented_sum, tf.reshape(counts, (-1, 1)))
    mu_expand= tf.gather(mu, unique_id)

    #calculate Lvar
    #[|µc − xi| − δv]^2
    distance= tf.norm(tf.subtract(mu_expand, prediction), axis= 1, ord= 1)
    distance= tf.subtract(distance, delta_v)
    distance= tf.clip_by_value(distance, 0., distance)
    distance= tf.square(distance)
    # 1/c c:1->C 1/N N:1->N PN ([|µc − xi| − δv]^2)
    Lvar= tf.unsorted_segment_sum(distance, unique_id, num_instances)
    Lvar= tf.div(Lvar, counts)
    Lvar= tf.reduce_sum(Lvar)
    Lvar= tf.divide(Lvar, tf.cast(num_instances, tf.float32))

    #calculate Ldist
    mu_interleaved_rep= tf.tile(mu, [num_instances, 1])
    mu_band_rep= tf.tile(mu, [1, num_instances])
    mu_band_rep= tf.reshape(my_band_rep, (num_instances**2, feature_dim))
    
    mu_diff= tf.subtract(mu_band_rep, mu_interleaved_rep)
    intermediate_tensor= tf.reduce_sum(tf.abs(mu_diff), axis= 1)
    zero_vector= tf.zeros(1, dtype= tf.float32)
    bool_mask= tf.not_equal(intermediate_tensor, zero_vector)
    mu_diff_bool= tf.boolean_mask(mu_diff, bool_mask)

    mu_norm= tf.norm(mu_diff_bool, axis= 1, ord= 1)
    mu_norm= tf.subtract(2. * delta_d, mu_norm)
    mu_norm= tf.clip_by_value(mu_norm, 0., mu_norm)
    mu_norm= tf.square(mu_norm)

    Ldist= tf.reduce_mean(mu_norm)
    Lreg= tf.reduce_mean(tf.norm(mu, axis= 1, ord= 1))

    param_scale= 1.

    Lvar= param_var * Lvar
    Ldist= param_dist * Ldist
    Lreg= param_reg * Lreg
    loss= param_scale * (Lvar + Ldist + Lreg)

    return loss, Lvar, Ldist, Lreg



def discriminative_loss(prediction, correct_label, feature_dim, image_shape,
                        delta_v, delta_d, param_var, param_dist, param_reg):

    print("correct_label:", correct_label.shape)
    print("prediction:", prediction.shape)
    print("feature_dim:", feature_dim)
    print("image_shape:", image_shape)

    
    def cond(label, batch, out_loss, out_var, out_dist, out_reg, i):
        return tf.less(i, tf.shape(batch)[0])
    
    def body(label, batch, out_loss, out_var, out_dist, out_reg, i):
        disc_loss, l_var, l_dist, l_reg= discriminative_loss_single(
            prediction[i], correct_label[i], feature_dim, image_shape,
            delta_v, delta_d, param_var, param_dist, param_reg
        )
        out_loss= out_loss.write(i, disc_loss)
        out_var= out_var.write(i, l_var)
        out_dist= out_dist.write(i, l_dist)
        out_reg= out_reg.write(i, l_reg)

        return label, batch, out_loss, out_var, out_dist, out_reg, i+1

    #tensorArray is a datastructure that supports dynamic writing
    output_loss= tf.TensorArray(dtype= tf.float32, size= 0, dynamic_size= True)
    output_var= tf.TensorArray(dtype= tf.float32, size= 0, dynamic_size= True)
    output_dist= tf.TensorArray(dtype= tf.float32, size= 0, dynamic_size= True)
    output_reg= tf.TensorArray(dtype= tf.float32, size= 0, dynamic_size= True)

    _, _, out_loss_op, out_var_op, out_dist_op, out_reg_op, _= tf.while_loop(cond, body, [correct_label, prediction, output_loss, output_var, output_dist, output_reg, 0])
    
    out_loss_op= out_loss_op.stack()
    out_var_op= out_var_op.stack()
    out_dist_op= out_dist_op.stack()
    out_reg_op= out_reg_op.stack()

    disc_loss= tf.reduce_mean(out_loss_op)
    l_var= tf.reduce_mean(out_var_op)
    l_dist= tf.reduce_mean(out_dist_op)
    l_reg= tf.reduce_mean(out_reg_op)

    return disc_loss, l_var, l_dist, l_reg

