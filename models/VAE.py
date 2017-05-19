from __future__ import absolute_import
from __future__ import division 
from __future__ import print_function

import numpy as np

import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

import pdb

""" Standard VAE: P(Z)P(X|Z) """

class VAE:
    def __init__(self, Z_DIM=2, ARCHITECTURE=[4,4], LEARNING_RATE=0.005, NONLINEARITY=tf.nn.relu,
    		 BATCH_SIZE=16,NUM_EPOCHS=75, Z_SAMPLES=1, verbose=1):

        self.Z_DIM = Z_DIM                                   # stochastic layer dimension       
    	self.ARCHITECTURE = ARCHITECTURE                     # number of hidden layers per network
    	self.NONLINEARITY = NONLINEARITY		     # activation functions	
    	self.lr = LEARNING_RATE 			     # learning rate
	self.BATCH_SIZE = BATCH_SIZE                         # batch size 
    	self.Z_SAMPLES = Z_SAMPLES 			     # number of monte-carlo samples
    	self.NUM_EPOCHS = NUM_EPOCHS                         # training epochs
    	self.LOGDIR = self._allocate_directory()             # logging directory
    	self.verbose = verbose				     # control output: 1 for ELBO, else accuracy

    def fit(self, Data):
    	self._process_data(Data)

    	# Step 1: define placeholders for input output
    	self._create_placeholders()

    	# Step 2: define weights and initialize networks
    	self._initialize_networks()

    	# Step 3: define the loss function
    	z_mean, z_log_var, z = self._sample_Z(self.x_batch)
    	KLz = self._gauss_kl(z_mean, tf.exp(z_log_var))
    	logpx = self._log_x_z(self.x_batch, z)
    	self.loss = tf.subtract(logpx , KLz, name='loss')

    	# Step 4: define optimizer
    	self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    	# Step 5: train and run
    	epoch, step = 0, 0
    	with tf.Session() as sess:
    	    sess.run(tf.global_variables_initializer())
    	    total_loss = 0
    	    writer = tf.summart.FileWriter(self.LOGDIR, sess)
    	    while epoch < self.NUM_EPOCHS:
    	    	batch = Data.next_batch_regular(self.BATCH_SIZE)
    	    	_, loss_batch = sess.run([self.optimizer, self.loss], 
    	    		feed_dict={self.x_batch: batch[0]})
    	    	total_loss += loss_batch
    	    	step = step + 1 
    	    	if Data._epochs_regular > epoch:
    	    	    epoch +=1
    	    	    print('Epoch: {}, Loss: {:5.3f}'.format(step, total_loss / step))
    	    	    total_loss, step = 0.0, 0
    	    writer.close()

    
    def _encode(self, x):
    	mean, log_var = self._forward_pass_Gauss(x, self.Qx_z)
    	return mean

    def _decode(self, z):
    	mean, log_var = self._forward_pass_Gauss(z, self.Pz_x)
    	return mean

    def _sample_Z(self, x):
    	""" Sample from Z with the reparamterization trick """
	mean, log_var = self._forward_pass_Gauss(x, self.Qx_z)
	eps = tf.random_normal([n_samples, self.Z_DIM], 0, 1, dtype=tf.float32)
	return mean, log_var, tf.add(mean, tf.multiply(tf.sqrt(tf.exp(log_var)), eps))

    
    def _forward_pass_Gauss(self, x, weights):
    	""" Forward pass through the network with weights as a dictionary """
	for i, neurons in enumerate(self.ARCHITECTURE):
	    weight_name, bias_name = 'W'+str(i), 'b'+str(i)
	    if i==0:
		h = self.NONLINEARITY(tf.add(tf.matmul(x, weights[weight_name]), [bias_name]))
	    else:
	        h = self.NONLINEARITY(tf.add(tf.matmul(h, self.weights[weight_name]), self.weights[bias_name]))
	mean = tf.add(tf.matmul(h, self.weights['Wmean']), self.weights['bmean'])
	log_var = tf.add(tf.matmul(h, self.weights['Wvar']), self.weights['bvar'])
	return mean, log_var


    def _log_x_z(self, x, z):
    	""" compute the likelihood of every element in x under p(x|z) """
	mean, log_var = self._forward_pass_Gauss(z, self.Pz_x)
	mvn = tf.contrib.distributions.MultivariateNormalDiag(loc=mean, scale_diag=tf.exp(log_var))
	return mvn.log_prob(x)


    def _initialize_networks(self):
    	self.Pz_x = self._init_Gauss_net(self.Z_DIM, self.ARCHITECTURE, self.X_DIM)
    	self.Qx_z = self._init_Gauss_net(self.X_DIM, self.ARCHITECTURE, self.Z_DIM)


    def _gauss_kl(self, mean, sigma):
	""" compute the KL-divergence of a Gaussian against N(0,1) """
	mean_0, sigma_0 = tf.zeros_like(mean), tf.ones_like(sigma)
	mvnQ = tf.contrib.distributions.MultivariateNormalDiag(loc=mean, scale_diag=sigma)
	prior = tf.contrib.distributions.MultivariateNormalDiag(loc=mean_0, scale_diag=sigma_0)
	return tf.contrib.distributions.kl(mvnQ, prior)

    def _init_Gauss_net(self, n_in, architecture, n_out):
    	""" initialize a network to parameterize a Gaussian """
    	weights = {}
    	for i, neurons in enumerate(architecture):
    	    weight_name, bias_name = 'W'+str(i), 'b'+str(i)
    	    if i == 0:
    	    	weights[weight_name] = tf.Variable(self._xavier_initializer(n_in, architecture[i]))
    	    else:
    	    	weights[weight_name] = tf.Variable(self._xavier_initializer(architecture[i-1], architecture[i]))
    	    weights[bias_name] = tf.Variable(tf.zeros(architecture[i]))
    	weights['Wmean'] = tf.Variable(self._xavier_initializer(architecture[-1], n_out))
    	weights['bmean'] = tf.Variable(tf.zeros(n_out))
    	weights['Wvar'] = tf.Variable(self._xavier_initializer(architecture[-1], n_out))
    	weights['bvar'] = tf.Variable(tf.zeros(n_out))
    	return weights


    def _xavier_initializer(self, fan_in, fan_out, constant=1): 
    	""" Xavier initialization of network weights"""
	low = -constant*np.sqrt(6.0/(fan_in + fan_out)) 
    	high = constant*np.sqrt(6.0/(fan_in + fan_out))
    	return tf.random_uniform((fan_in, fan_out), 
        	                  minval=low, maxval=high, 
            	                  dtype=tf.float32)

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

    