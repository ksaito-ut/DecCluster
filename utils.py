import tensorflow as tf
import numpy as np
from sklearn import preprocessing
# Model construction utilities below adapted from
# https://www.tensorflow.org/versions/r0.8/tutorials/mnist/pros/index.html#deep-mnist-for-experts
def kl_divergence(p,q):
    y = tf.div(p,q)+1e-6
    return tf.reduce_mean(-tf.nn.softmax_cross_entropy_with_logits(p, y))
def cluster_acc(Y_pred, Y):
  from sklearn.utils.linear_assignment_ import linear_assignment
  assert Y_pred.size == Y.size
  D = max(Y_pred.max(), Y.max())+1
  w = np.zeros((D,D), dtype=np.int32)
  for i in xrange(Y_pred.size):
    w[Y_pred[i], Y[i]] += 1
  ind = linear_assignment(w.max() - w)
  return sum([w[i,j] for i,j in ind])*1.0/Y_pred.size, w
def squared_dist(A,B): 
    expanded_a = tf.expand_dims(A, 1)
    expanded_b = tf.expand_dims(B, 0)
    distances = tf.reduce_sum(tf.squared_difference(expanded_a, expanded_b), 2)
    return distances
def calc_q(feature,cluster,cluster_num=10,batch_size=128):
    q = 1 + squared_dist(feature,cluster)
    q = 1./q
    q_sum = tf.reduce_sum(1 + squared_dist(feature,cluster),0)
    q_sum = 1./q_sum
    q = q/q_sum
    return q
def calc_p(q,cluster_num=10,batch_size=128):
    f = tf.reduce_sum(q,1)
    f = 1./f
    f = tf.reshape(f,[cluster_num,1])
    div_value = tf.matmul(q*q,f)
    div_value = tf.reduce_sum(div_value,0)
    p = tf.matmul(q*q,f)
    p = p / div_value
    return q
def weight_variable(shape,stddev=0.1,name=None,train=True):
    #initial = tf.random_normal(shape, stddev=0.1, dtype=tf.float32)#
    initial = tf.truncated_normal(shape, stddev=stddev) #default 0.1
    if name:
        #W = tf.get_variable(name, shape=shape,
#           initializer=tf.contrib.layers.xavier_initializer())
        return tf.Variable(initial,name=name,trainable=train)
    else:
        return tf.Variable(initial)
def bias_variable(shape,init=0.1,name=None):
    initial = tf.constant(init, shape=shape)
    if name:
        return tf.Variable(initial,name=name)
    else:
        return tf.Variable(initial)
def shuffle_aligned_list(data):
    """Shuffle arrays in a list by shuffling each array identically."""
    num = data[0].shape[0]
    p = np.random.permutation(num)
    return [d[p] for d in data]
def batch_generator(data, batch_size, shuffle=True):
    """Generate batches of data.
    
    Given a list of array-like objects, generate batches of a given
    size by yielding a list of array-like objects corresponding to the
    same slice of each input.
    """
    if shuffle:
        data = shuffle_aligned_list(data)
    batch_count = 0
    while True:
        if batch_count * batch_size + batch_size >= len(data[0]):
            batch_count = 0
            if shuffle:
                data = shuffle_aligned_list(data)
        start = batch_count * batch_size
        end = start + batch_size
        batch_count += 1
        yield [d[start:end] for d in data]
