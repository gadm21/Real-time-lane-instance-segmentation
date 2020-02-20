
import tensorflow as tf

# discriminative loss for instance segmentation

def discriminative_loss_single(
    prediction, #inference of network
    correct_label, #instance label
    feature_dim, #feature dimension of prediction
    label_shape, #shape of label
    delta_v, #cut-off vairance distance    
    delta_d, #cut-off cluster distance
    param_v, #weight for intra cluster distance 
    param_dist, #weight for inter cluster distance
    param_reg #weight regularization
    ):

    reshaped_label= tf.reshape(correct_label, [label_shape[1]* label_shape[0]])
    reshaped_prediction= tf.reshape(prediction, [label_shape[1]* label_shape[0], feature_dim])

    #calculate instance nums
    unique_labels, unique_id, counts= tf.unique_with_counts(reshaped_label)
    counts= tf.cast(counts, tf.float32)
    num_instances= tf.size(unique_labels)

    #calculate instance pixel embedding mean vector
    segmented_sum= tf.unsorted_segment_sum(reshaped_prediction, unique_id, num_instances)
    mu= tf.div()