from __future__ import absolute_import
from __future__ import division 
from __future__ import print_function

import numpy as np

import utils.dgm as dgm
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

import pdb

""" Standard VAE: P(Z)P(X|Z) """

class VAE:
    def __init__(self, Z_DIM=2, NUM_HIDDEN=[4,4], LEARNING_RATE=0.005, NONLINEARITY=tf.nn.relu,
    		 BATCH_SIZE=16,NUM_EPOCHS=75, Z_SAMPLES=1):

        self.Z_DIM = Z_DIM                                   # stochastic layer dimension       
    	self.NUM_HIDDEN = NUM_HIDDEN                         # number of hidden layers per network
    	self.NONLINEARITY = NONLINEARITY		     # activation functions	
    	self.lr = LEARNING_RATE 			     # learning rate
	self.BATCH_SIZE = BATCH_SIZE                         # batch size 
    	self.Z_SAMPLES = Z_SAMPLES 			     # number of monte-carlo samples
    	self.NUM_EPOCHS = NUM_EPOCHS                         # training epochs
    	self.LOGDIR = self._allocate_directory()             # logging directory

    def fit(self, Data):
    	self._process_data(Data)

    	# define placeholders for input output
    	self._create_placeholders()
    	# define weights and initialize networks
    	self._initialize_networks()
    	# define the loss function
    	self.loss = -self._compute_ELBO(self.x_batch)
	test_elbo = self._compute_ELBO(self.x_test)
	train_elbo = self._compute_ELBO(self.x_train)
    	# define optimizer
    	self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
    	# run and train
    	epoch, step = 0, 0
    	with tf.Session() as sess:
    	    sess.run(tf.global_variables_initializer())
    	    total_loss = 0
    	    writer = tf.summary.FileWriter(self.LOGDIR, sess.graph)
    	    while epoch < self.NUM_EPOCHS:
    	    	batch = Data.next_batch_regular(self.BATCH_SIZE)
    	    	_, loss_batch = sess.run([self.optimizer, self.loss], 
    	    		feed_dict={self.x_batch: batch[0]})
    	    	total_loss += loss_batch
    	    	step = step + 1 
    	    	if Data._epochs_regular > epoch:
		    trainELBO, testELBO = sess.run([train_elbo, test_elbo],
						   feed_dict={self.x_train:Data.data['x_train'],
							      self.x_test:Data.data['x_test']})
    	    	    print('Epoch: {}, Train ELBO: {:5.3f}, Test ELBO: {:5.3f}'.format(epoch, trainELBO, testELBO))
    	    	    total_loss, step, epoch = 0.0, 0, epoch + 1
    	    writer.close()

    
    def _encode(self, x):
    	mean, log_var = dgm._forward_pass_Gauss(x, self.Qx_z, self.NUM_HIDDEN, self.NONLINEARITY)
    	return mean

    def _decode(self, z):
    	mean, log_var = dgm._forward_pass_Gauss(z, self.Pz_x, self.NUM_HIDDEN, self.NONLINEARITY)
    	return mean

    def _sample_Z(self, x, n_samples=1):
    	""" Sample from Z with the reparamterization trick """
	mean, log_var = dgm._forward_pass_Gauss(x, self.Qx_z, self.NUM_HIDDEN, self.NONLINEARITY)
	eps = tf.random_normal([n_samples, self.Z_DIM], 0, 1, dtype=tf.float32)
	return mean, log_var, tf.add(mean, tf.multiply(tf.sqrt(tf.exp(log_var)), eps))

    def _compute_ELBO(self, x):
    	z_mean, z_log_var, z = self._sample_Z(x)
    	KLz = dgm._gauss_kl(z_mean, tf.exp(z_log_var))
    	logpx = self._log_x_z(x, z),
	total_elbo = tf.subtract(logpx, KLz)
        return tf.reduce_mean(total_elbo)

    
    def _log_x_z(self, x, z):
    	""" compute the likelihood of every element in x under p(x|z) """
	mean, log_var = dgm._forward_pass_Gauss(z, self.Pz_x, self.NUM_HIDDEN, self.NONLINEARITY)
	mvn = tf.contrib.distributions.MultivariateNormalDiag(loc=mean, scale_diag=tf.exp(log_var))
	return mvn.log_prob(x)


    def _initialize_networks(self):
    	self.Pz_x = dgm._init_Gauss_net(self.Z_DIM, self.NUM_HIDDEN, self.X_DIM)
    	self.Qx_z = dgm._init_Gauss_net(self.X_DIM, self.NUM_HIDDEN, self.Z_DIM)



    def _process_data(self, data):
    	""" Extract relevant information from data_gen """
    	self.dataset = data.NAME                                 # name of dataset
    	self.N = data.N                                          # number of examples
    	self.TRAINING_SIZE = data.TRAIN_SIZE   			 # training set size
	self.TEST_SIZE = data.TEST_SIZE                          # test set size
	self.X_DIM = data.INPUT_DIM            			 # input dimension     
	self.NUM_CLASSES = data.NUM_CLASSES                      # number of classes
	

    def _create_placeholders(self):
    	self.x_train = tf.placeholder(tf.float32, shape=[self.TRAINING_SIZE, self.X_DIM], name='x_train')
    	self.y_train = tf.placeholder(tf.float32, shape=[self.TRAINING_SIZE, self.NUM_CLASSES], name='y_train')
    	self.x_test = tf.placeholder(tf.float32, shape=[self.TEST_SIZE, self.X_DIM], name='x_test')
    	self.y_test = tf.placeholder(tf.float32, shape=[self.TEST_SIZE, self.NUM_CLASSES], name='y_test')
    	self.x_batch = tf.placeholder(tf.float32, shape=[self.BATCH_SIZE, self.X_DIM], name='x_batch')
    	self.y_batch = tf.placeholder(tf.float32, shape=[self.BATCH_SIZE, self.NUM_CLASSES], name='y_batch')

    def _allocate_directory(self):
	return 'VAE_graphs/default/'    
