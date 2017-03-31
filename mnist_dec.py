import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
from utils import *
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
# Process MNIST
mnist_train = (mnist.train.images > 0).reshape(55000, 28*28).astype(np.uint8) * 255
mnist_test = (mnist.test.images > 0).reshape(10000, 28*28).astype(np.uint8) * 255
batch_size = 128

class MNISTModel(object):
    """Simple MNIST domain adaptation model."""
    def __init__(self):
        self._build_model()
    
    def _build_model(self,cluster_num=10):        
        self.X = tf.placeholder(tf.uint8, [None, 28*28])
        X_input = tf.cast(self.X,tf.float32) / 255.
        # CNN model for feature extraction
        #with tf.device('/gpu:0'):
        with tf.variable_scope('feature_extractor'):
            with slim.arg_scope([slim.fully_connected],activation_fn=tf.nn.relu,weights_initializer=tf.contrib.layers.xavier_initializer()):
                net = slim.fully_connected(X_input,392,scope='enc1')
                net = slim.fully_connected(net,196,scope='enc2')
                net = slim.fully_connected(net,100,scope='enc3')
                self.feature = net
                net = slim.fully_connected(net,196,scope='enc4')
                net = slim.fully_connected(net,392,scope='enc5')
                net = slim.fully_connected(net,784,scope='enc6',activation_fn =None)
                self.reconst_loss = tf.square(X_input-net)
        with tf.variable_scope('clustering'):
            myu = tf.Variable(tf.random_normal([cluster_num, 100], stddev=0.1),name="myu")
            vec_q = calc_q(self.feature,myu)
            self.q = vec_q
            self.p = calc_p(vec_q)
            self.kl_loss = kl_divergence(self.p,self.p)
# Build the model graph
graph = tf.get_default_graph()
with graph.as_default():
    model = MNISTModel()    
    learning_rate = tf.placeholder(tf.float32, [])          
    reconst_loss = tf.reduce_mean(model.reconst_loss)
    kl_loss = model.kl_loss
    total_loss = reconst_loss + 0.01*kl_loss
    regular_train_op = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(total_loss)    
    reconst_train_op = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(reconst_loss)    
    # Evaluation

# Params
num_steps = 1000

def train_and_evaluate(training_mode, graph, model, verbose=True,mode='kl'):
    """Helper to run the model with different training modes."""
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    #sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    with tf.Session(graph=graph,config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        tf.initialize_all_variables().run()
        # Batch generators
        gen_source_batch = batch_generator(
            [mnist_train, mnist.train.labels], batch_size)
        gen_test_batch = batch_generator(
            [mnist_test,np.squeeze(mnist.test.labels)], batch_size)

        # Training loop
        for i in range(num_steps):
            lr = 0.01 
            # Training step
            X, y = gen_source_batch.next()
            if mode =='kl':
                _, r_loss,kl,vec = sess.run([regular_train_op,reconst_loss,kl_loss,model.p],
                                            feed_dict={model.X: X,learning_rate:lr})
            else:
                _, r_loss,kl,vec = sess.run([reconst_train_op,reconst_loss,kl_loss,model.p],
                                            feed_dict={model.X: X,learning_rate:lr})
            if verbose and i % 100 == 0:
                print vec[0,:]
                print 'reconst_loss: %f kl: %f' % \
                    (r_loss,kl)


        steps = mnist_test.shape[0]/batch_size
        prediction = np.zeros((0,10))
        label = np.zeros((0,10))
        while steps:
            X, y = gen_test_batch.next()
            vec = sess.run(model.q,
                                     feed_dict={model.X: X,learning_rate:lr})
            prediction = np.r_[prediction,vec]
            label = np.r_[label,y]
            steps -= 1
        prediction = np.argmax(prediction,axis=1)
        label = np.argmax(label,axis=1)
        acc,_ = cluster_acc(prediction,label)
        print 'clusterinig acc:',acc
if __name__ == '__main__':
    train_and_evaluate('target', graph, model,mode='kl')
