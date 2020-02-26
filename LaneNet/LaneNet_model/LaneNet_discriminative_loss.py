
#Discriminative loss for instance segmentation

import tensorflow as tf



'''
The 2nd argument to cond() & body() is prdiction not batch, but it called and treated as batch inside the function!! 
How is body using prediction[i]? 
Is prediction related to batch ?

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



def discriminative_loss(prediction, correct_label, feature_dim, image_shape,
                        delta_v, delta_d, param_var, param_dist, param_reg):

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
    output_loss= tf.TensorArray(dtype= tf.float32, dynamic_size= True)
    output_var= tf.TensorArray(dtype= tf.float32, dynamic_size= True)
    output_dist= tf.TensorArray(dtype= tf.float32, dynamic_size= True)
    output_reg= tf.TensorArray(dtype= tf.float32, dynamic_size= True)

    _, _, out_loss_op, out_var_op, out_dist_op, out_reg_op, _= tf.while_loop(cond, body, [correct_label, prediction, output_loss, output_var, output_dist, output_reg])

    out_loss_op= out_loss_op.stack()
    out_var_op= out_var_op.stack()
    out_dist_op= out_dist_op.stack()
    out_reg_op= out_reg_op.stack()

    disc_loss= tf.reduce_mean(out_loss_op)
    l_var= tf.reduce_mean(out_var_op)
    l_dist= tf.reduce_mean(out_dist_op)
    l_reg= tf.reduce_mean(out_reg_op)

    return disc_loss, l_var, l_dist, l_reg