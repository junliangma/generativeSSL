from __future__ import absolute_import
from __future__ import division 
from __future__ import print_function

import numpy as np

import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

import pdb

### Initial class build (moons data in mind) ###

class generativeSSL:
    """ Class defining our generative model """
    def __init__(self, Z_DIM=2, LEARNING_RATE=0.01, NUM_HIDDEN=[4], ALHPA=0.1, LABELED_BATCH_SIZE=100,
    			 UNLABELED_BATCH_SIZE=10, NUM_STEPS=1000, Z_SAMPLES=1)
    	## Step 1: define the placeholders for input and output
    	self.z_dim = Z_DIM                                  # stochastic inputs dimension       
    	self.n_h = NUM_HIDDEN                               # number of hidden layers per network
    	self.lr = LEARNING_RATE 			    # learning rate
    	self.alpha = ALHPA 				    # weighting for additional term
    	self.Z_SAMPLES = Z_SAMPLES 			    # number of monte-carlo samples
    	self.NUM_STEPS = NUM_STEPS                          # training steps
    	self.LOGDIR = self._allocate_directory()            # logging directory
    
    def fit(self, data):
    	self._process_data(data)
    	
    	# Step 1: define the placeholders for input and output
    	self._create_placeholders()
    
       	## Step 2: define weights - setup all networks
        self._intialize_networks()
        
        ## Step 3: define the loss function
        L_l = tf.reduce_sum(self._labeled_loss(self.x_labeled, self.labels))
        L_u = tf.reduce_sum(self._unlabeled_loss(self.x_unlabeled))
        L_e = self._qxy_loss(self.x_labeled, self.y_labeled)
        self.loss = tf.add_n([L_l , L_u , self.alpha*L_e], name='loss')
        
        ## Step 4: define optimizer
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        
        ## Step 5: initialize session and train
        SKIP_STEP = 10
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            total_loss = 0.0 # we use this to calculate the average loss in the last SKIP_STEP steps
            writer = tf.summary.FileWriter(LOGDIR, sess.graph)
            for index in xrange(NUM_STEPS):
                batch = data.next()
            	_, loss_batch = sess.run([optimizer, loss], 
            			     	 feed_dict={self.x_labeled: batch[0], 
            		    	 	 self.x_unlabeled: batch[1],
            		     		 self.labels: batch[2]})
                total_loss += loss_batch
                if (index + 1) % SKIP_STEP == 0:
        	    print('Average loss at step {}: {:5.1f}'.format(index, total_loss / SKIP_STEP))
        	    total_loss = 0.0
	    writer.close()

    

    def _process_data(self, data):
    	""" Extract relevant information from data_gen """
    	self.TRAINING_SIZE = data.TRAIN_SIZE   			 # training set size
	self.NUM_LABELED =data.NUM_LABELED     			 # labeled instances
	self.X_DIM = data.INPUT_DIM            			 # input dimension     
	self.NUM_CLASSES = data.NUM_CLASSES                      # number of classes
	self.LABELED_BATCH_SIZE = data.LABELED_BATCH_SIZE        # labeled batch size 
	self.UNLABELED_BATCH_SIZE = data.UNLABELED_BATCH_SIZE    # labeled batch size 
	self.alpha = self.alpha / self.NUM_LABELED               # weighting for additional term


    def _create_placeholders(self):
 	""" Create input/output placeholders """
	self.x_labeled = tf.placeholder(tf.int32, shape=[self.LABELED_BATCH_SIZE, self.X_DIM], name='Labeled input')
    	self.x_unlabeled = tf.placeholder(tf.int32, shape=[self.UNLABELED_BATCH_SIZE, self.X_DIM], name='Unlabeled input')
    	self.labels = tf.placeholder(tf.int32, shape=[self.LABELED_BATCH_SIZE, self.NUM_CLASSES], name='Labels')
    	


    def _initialize_networks(self):
    	""" Initialize all model networks """
    	self.Pz_x = self._init_Gauss_net(self.Z_DIM, self.NUM_HIDDEN, self.X_DIM)
    	self.Pzx_y = self._init_Cat_net(self.Z_DIM+self.X_DIM, self.NUM_HIDDEN, self.NUM_CLASSES)
    	self.Qxy_z = self._init_Gauss_net(self.X_DIM+self.NUM_CLASSES, self.NUM_HIDDEN, self.Z_DIM)
    	self.Qx_y = self._init_Cat_net(self.X_DIM, self.NUM_HIDDEN, self.NUM_CLASSES)

    
    def _xavier_initializer(self, fan_in, fan_out, constant=1): 
    	""" Xavier initialization of network weights"""
	low = -constant*np.sqrt(6.0/(fan_in + fan_out)) 
    	high = constant*np.sqrt(6.0/(fan_in + fan_out))
    	return tf.random_uniform((fan_in, fan_out), 
        	                  minval=low, maxval=high, 
            	                  dtype=tf.float32)


    def _init_Gauss_net(self, n_in, n_hidden, n_out):
	""" Initialize the weights of a 2-layer network parameterizeing a Gaussian """
	W1 = tf.Variable(self._xavier_initializer(n_in, n_hidden))
	W2_mean = tf.Variable(self._xavier_initializer(n_hidden, n_out))
	W2_var = tf.Variable(self._xavier_initializer(n_hidden, n_out))
	b1 = tf.Variable(tf.zeros([n_hidden]))
	b2_mean = tf.Variable(tf.zeros([n_out]))
	b2_var = tf.Variable(tf.zeros([n_out]))
	return {'W_in':W1, 'W_out_mean':W2_mean, 'W_out_var':W2_var, 
	        'bias_in':b1, 'bias_out_mean':b2_mean, 'bias_out_var':b2_var}


    def _init_Cat_net(self, n_in, n_hidden, n_out):
	""" Initialize the weights of a 2-layer network parameterizeing a Categorical """
	W1 = tf.Variable(self._xavier_initializer(n_in, n_hidden))
	W2 = tf.Variable(self._xavier_initializer(n_hidden, n_out))
	b1 = tf.Variable(tf.zeros([n_hidden]))
	b2 = tf.Variable(tf.zeros([n_out]))
	return {'W_in':W1, 'W_out':W2, 'bias_in':b1, 'bias_out':b2}

    def _forward_pass_Gauss(self, input, weights):
	""" Forward pass through the network with weights as a dictionary """
	h = tf.nn.softplus(tf.add(tf.matmul(weights['W_in'], input), weights['bias_in']))
	mean = tf.add(tf.matmul(weights['W_out_mean'], input), weights['bias_out_mean'])
	sigma = tf.nn.softplus(tf.add(tf.matmul(weights['W_out_var'], input), weights['bias_out_var']))
	return (mean, sigma)


    def _forward_pass_Cat(self, input, weights):
	""" Forward pass through the network with weights as a dictionary """
	h = tf.nn.softplus(tf.add(tf.matmul(weights['W_in'], input), weights['bias_in']))
	out = tf.nn.softmax(add(tf.matmul(weights['W_out'], input), weights['bias_out']))
	return (out)


    def _sample_Z(self, x, y, n_samples):
	""" Sample from Z with the reparamterization trick """
	h = tf.concat([x, y], axis=0)
	q_mean, q_sigma = forward_pass_Gauss(h, self.Qxy_z)
	eps = tf.random_normal([n_samples, self.Z_DIM], 0, 1, dtype=float32)
	return (q_mean, q_sigma, tf.add(q_mean, tf.mul(tf.sqrt(q_sigma)), eps))


    def _labeled_loss(self, x, y):
	""" Compute necessary terms for labeled loss (per data point) """
	q_mean, q_sigma, z = sample_Z(x, y, self.Z_SAMPLES)
	logpx = self._compute_logpx(x, z)
	logpy = self._compute_logpy(y, x, z)
	klz = self._gauss_kl(q_mean, q_sigma)
	return tf.add_n([logpx , logpy , -klz])

    def _unlabeled_loss(self, x):
	""" Compute necessary terms for unlabeled loss (per data point) """
	weights = self._forward_pass_Cat(x, self.Qx_y)
	EL_l = 0 
	for i in range(NUM_CLASSES):
	    y = self._generate_class(i, x.get_shape()[0])
	    EL_l += tf.multiply(weights[:,i], self.labeled_loss(x, y))
	ent_qy = -tf.reduce_sum(tf.multiply(weights, tf.log(weights)))
	return tf.add(EL_l, ent_qy)


    def _qxy_loss(self, x, y):
	y_ = self._forward_pass_Cat(x, self.Qx_y)
	return tf.reduce_mean(tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))


    def _compute_logpx(self, x, z):
	""" compute the likelihood of every element in x under p(x|z) """
	mean, sigma = self._forward_pass_Gauss(z,self.Pz_x)
	mvn = tf.contrib.distributions.MultiVariateNormalDiag(loc=mean, scale_diag=sigma)
	return mvn.prob(x)


    def _compute_logpy(self, y, x, z):
	""" compute the likelihood of every element in y under p(y|x,z) """
	h = tf.concat([x,z], axis=0)
	mean, sigma = self._forward_pass_Gauss(h, self.Pzx_y)
	mvn = tf.contrib.distributions.MultiVariateNormalDiag(loc=mean, scale_diag=sigma)
	return mvn.prob(y)

    def _gauss_kl(self, mean, sigma):
	""" compute the KL-divergence of a Gaussian against N(0,1) """
	mean_0, sigma_0 = tf.zeros_like(mean), tf.ones_like(sigma)
	mvnQ = tf.contrib.distributions.MultiVariateNormalDiag(loc=mean, scale_diag=sigma)
	prior = tf.contrib.distributions.MultiVariateNormalDiag(loc=mean_0, scale_diag=sigma_0)
	return tf.contrib.distributions.kl(mvnQ, prior)


    def _generate_class(self, k, num):
	""" create one-hot encoding of class k with length num """
	y = np.zeros(shape=(num, self.NUM_CLASSES))
	y[:,k] = 1
	return tf.constant(y)

    def _allocate_directory(self):
	print 'NOT PROPERLY IMPLEMENTED'
	return 'graphs/default/'



