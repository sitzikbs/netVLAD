import tensorflow as tf

def VLAD_pooling(inputs,
              k_centers,
              scope,
              use_xavier=True,
              stddev=1e-3):
    """ VLAD orderless pooling - based on netVLAD paper:
  title={NetVLAD: CNN architecture for weakly supervised place recognition},
  author={Arandjelovic, Relja and Gronat, Petr and Torii, Akihiko and Pajdla, Tomas and Sivic, Josef},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={5297--5307},
  year={2016}

    Args:
      inputs: 4-D tensor BxHxWxC
      k_centers: scalar number of cluster centers

    Returns:
      Variable tensor
    """

    num_batches = inputs.get_shape()[0].value
    #num_feature_maps = inputs.get_shape()[1].value
    num_features = inputs.get_shape()[2].value

    #Initialize the variables for learning w,b,c - Random initialization
    if use_xavier:
        initializer = tf.contrib.layers.xavier_initializer()
    else:
        initializer = tf.truncated_normal_initializer(stddev=stddev)

    with tf.variable_scope(scope) as sc:
        w = tf.get_variable('weights',
                            shape=[k_centers, num_features],
                            initializer=initializer)
        b = tf.get_variable('biases',
                            shape=[k_centers, 1],
                            initializer=initializer)
        c = tf.get_variable('centers',
                            shape=[k_centers, num_features],
                            initializer=initializer)

        # # Initialize with pre-computed VLAD as centers (using kmeans on a trained model - yields no improvement over random init
        # alpha = 1000
        # PATH_TO_CENTERS = 'path_to_saved_centers'
        # filename = 'VLAD_centers_' + str(k_centers) + '.npy'
        # centers = np.load(PATH_TO_CENTERS + filename)
        # c = tf.Variable( centers, name='cetners')
        # w = tf.Variable(2 * alpha * centers, name='weights' )
        # b = tf.Variable(-alpha * tf.pow(tf.norm(centers, axis=1),2), name = 'biases')

        #Pooling
        for k in range(k_centers):

            wk = tf.expand_dims(tf.tile(tf.expand_dims(w[k, :],0),multiples=[num_batches,1]),[-1])
            Wx_b = tf.matmul(inputs, wk)  + b[k]
            a = tf.nn.softmax( Wx_b )
            if k == 0:
                outputs =  tf.reduce_sum(tf.multiply(a, (inputs - tf.slice(c, [k, 0], [1, num_features]))), axis=1)
                outputs = tf.expand_dims(outputs,1)
            else:
                outputs = tf.concat([outputs, tf.expand_dims(tf.reduce_sum(tf.multiply(a, (inputs - tf.slice(c,[k, 0], [1, num_features]))), axis=1),1)], 1)

        outputs = tf.nn.l2_normalize(outputs,dim=2) #intra-normalization
        outputs = tf.reshape(outputs,[num_batches,-1])
        outputs = tf.nn.l2_normalize(outputs,dim=1) #l2 normalization
        return outputs
