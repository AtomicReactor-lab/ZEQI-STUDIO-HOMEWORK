import tensorflow as tf

def get_bias(shape):
    bias = tf.get_Variable(tf.zeros(shape))
    return bias

def get_weight(shape, regularizer):
    w = tf.get_Variable(tf.truncated_normal(shape,stddev=0.1))
    if regularizer != None:
        tf.add_to_collection('losses', tf.contrib_layers.l2_regularizer(regularizer))
    return w

def conv2d(x,w):
    return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='VALID')

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')