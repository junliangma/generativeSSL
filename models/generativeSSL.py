from __future__ import absolute_import
from __future__ import division 
from __future__ import print_function

import sys, os, pdb

import numpy as np
import utils.dgm as dgm 

import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector


""" Generative models for labels with stochastic inputs: P(Z)P(X|Z)P(Y|X,Z) """

class generativeSSL:
   
    def __init__(self, Z_DIM=2, LEARNING_RATE=0.005, NUM_HIDDEN=[4], ALPHA=0.1, TYPE_PX='Gaussian', NONLINEARITY=tf.nn.relu, 
                 LABELED_BATCH_SIZE=16, UNLABELED_BATCH_SIZE=128, NUM_EPOCHS=75, Z_SAMPLES=1, BINARIZE=False, verbose=1):
    	## Step 1: define the placeholders for input and output
    	self.Z_DIM = Z_DIM                                   # stochastic inputs dimension       
    	self.NUM_HIDDEN = NUM_HIDDEN                         # (list) number of hidden layers/neurons per network
    	self.NONLINEARITY = NONLINEARITY		     # activation functions	
 	self.TYPE_PX = TYPE_PX				     # Distribution of X (Gaussian, Bernoulli)	
    	self.lr = LEARNING_RATE 			     # learning rate
    	self.LABELED_BATCH_SIZE = LABELED_BATCH_SIZE         # labeled batch size 
	self.UNLABELED_BATCH_SIZE = UNLABELED_BATCH_SIZE     # labeled batch size 
    	self.alpha = ALPHA 				     # weighting for additional term
    	self.Z_SAMPLES = Z_SAMPLES 			     # number of monte-carlo samples
    	self.NUM_EPOCHS = NUM_EPOCHS                         # training epochs
	self.BINARIZE = BINARIZE                             # sample inputs from Bernoulli distribution if true 
    	self.LOGDIR = self._allocate_directory()             # logging directory
    	self.verbose = verbose				     # control output: 0-ELBO, 1-accuracy, 2-Q-accuracy
    

    def fit(self, Data):
    	self._process_data(Data)
    	
	self._create_placeholders() 
        self._initialize_networks()
        
        ## define loss function
	self._compute_loss_weights()
        L_l = tf.reduce_sum(self._labeled_loss(self.x_labeled, self.labels))
        L_u = tf.reduce_sum(self._unlabeled_loss(self.x_unlabeled))
        L_e = self._qxy_loss(self.x_labeled, self.labels)
        self.loss = -tf.add_n([self.labeled_weight*L_l , self.unlabeled_weight*L_u , self.alpha*L_e], name='loss')
        
        ## define optimizer
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
	
	## compute accuracies
	train_acc, train_acc_q= self.compute_acc(self.x_train, self.y_train)
	test_acc, test_acc_q = self.compute_acc(self.x_test, self.y_test)
	
        ## initialize session and train
        SKIP_STEP, epoch, step = 50, 0, 0
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            total_loss, l_l, l_u, l_e = 0.0, 0.0, 0.0, 0.0
            writer = tf.summary.FileWriter(self.LOGDIR, sess.graph)
            while epoch < self.NUM_EPOCHS:
                x_labeled, labels, x_unlabeled, _ = Data.next_batch(self.LABELED_BATCH_SIZE, self.UNLABELED_BATCH_SIZE)
		if self.BINARIZE == True:
	 	    x_labeled, x_unlabeled = self._binarize(x_labeled), self._binarize(x_unlabeled)
            	_, loss_batch, l_lb, l_ub, l_eb = sess.run([self.optimizer, self.loss, L_l, L_u, L_e], 
            			     	     		    feed_dict={self.x_labeled: x_labeled, 
		           		    	 		       self.labels: labels,
		  	            		     		       self.x_unlabeled: x_unlabeled})
                total_loss, l_l, l_u, l_e, step = total_loss+loss_batch, l_l+l_lb, l_u+l_ub, l_e+l_eb, step+1
                if Data._epochs_unlabeled > epoch:
		    epoch += 1
		    if self.verbose == 0:
		    	self._hook_loss(epoch, step, total_loss, l_l, l_u, l_e)
        	        total_loss, l_l, l_u, l_e, step = 0.0, 0.0, 0.0, 0.0, 0
        	    
		    elif self.verbose == 1:
			x_train = Data.data['x_train']
			x_test = Data.data['x_test']
			    
		        acc_train, acc_test,  = sess.run([train_acc, test_acc],
						         feed_dict = {self.x_train:x_train,
						     	              self.y_train:Data.data['y_train'],
								      self.x_test:x_test,
								      self.y_test:Data.data['y_test']})
		        print('At epoch {}: Training: {:5.3f}, Test: {:5.3f}'.format(epoch, acc_train, acc_test))
        	    
		    elif self.verbose == 2:
		        acc_train, acc_test,  = sess.run([train_acc_q, test_acc_q],
						         feed_dict = {self.x_train:x_train,
						     	              self.y_train:Data.data['y_train'],
								      self.x_test:x_test,
								      self.y_test:Data.data['y_test']})
		        print('At epoch {}: Training: {:5.3f}, Test: {:5.3f}'.format(epoch, acc_train, acc_test))
	    writer.close()


    def predict(self, x, n_iters=100):
	y_ = dgm._forward_pass_Cat(x, self.Qx_y, self.NUM_HIDDEN, self.NONLINEARITY)
	yq = y_
	y_ = tf.one_hot(tf.argmax(y_, axis=1), self.NUM_CLASSES)
	y_samps = tf.expand_dims(y_, axis=2)
	for i in range(n_iters):
	    _, _, z = self._sample_Z(x, y_, self.Z_SAMPLES)
	    h = tf.concat([x, z], axis=1)
	    y_ = dgm._forward_pass_Cat(h, self.Pzx_y, self.NUM_HIDDEN, self.NONLINEARITY)
	    y_samps = tf.concat([y_samps, tf.expand_dims(y_, axis=2)], axis=2)
	    y_ = tf.one_hot(tf.argmax(y_, axis=1), self.NUM_CLASSES)
	return tf.reduce_mean(y_samps, axis=2), yq



    def _sample_Z(self, x, y, n_samples):
	""" Sample from Z with the reparamterization trick """
	h = tf.concat([x, y], axis=1)
	mean, log_var = dgm._forward_pass_Gauss(h, self.Qxy_z, self.NUM_HIDDEN, self.NONLINEARITY)
	eps = tf.random_normal([n_samples, self.Z_DIM], 0, 1, dtype=tf.float32)
	return mean, log_var, tf.add(mean, tf.multiply(tf.sqrt(tf.exp(log_var)), eps))


    def _labeled_loss(self, x, y):
	""" Compute necessary terms for labeled loss (per data point) """
	q_mean, q_log_var, z = self._sample_Z(x, y, self.Z_SAMPLES)
	logpx = self._compute_logpx(x, z)
	logpy = self._compute_logpy(y, x, z)
	klz = dgm._gauss_kl(q_mean, tf.exp(q_log_var))
	return tf.add_n([logpx , logpy , -klz])


    def _unlabeled_loss(self, x):
	""" Compute necessary terms for unlabeled loss (per data point) """
	weights = dgm._forward_pass_Cat(x, self.Qx_y, self.NUM_HIDDEN, self.NONLINEARITY)
	EL_l = 0 
	for i in range(self.NUM_CLASSES):
	    y = self._generate_class(i, x.get_shape()[0])
	    EL_l += tf.multiply(weights[:,i], self._labeled_loss(x, y))
	ent_qy = -tf.reduce_sum(tf.multiply(weights, tf.log(weights)), axis=1)
	return tf.add(EL_l, ent_qy)


    def _qxy_loss(self, x, y):
	y_ = dgm._forward_pass_Cat_logits(x, self.Qx_y, self.NUM_HIDDEN, self.NONLINEARITY)
	return -tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_))


    def _compute_logpx(self, x, z):
	""" compute the likelihood of every element in x under p(x|z) """
	if self.TYPE_PX == 'Gaussian':
	    mean, log_var = dgm._forward_pass_Gauss(z,self.Pz_x, self.NUM_HIDDEN, self.NONLINEARITY)
	    mvn = tf.contrib.distributions.MultivariateNormalDiag(loc=mean, scale_diag=tf.exp(log_var))
	    return mvn.log_prob(x)
	elif self.TYPE_PX == 'Bernoulli':
	    pi = dgm._forward_pass_Bernoulli(z, self.Pz_x, self.NUM_HIDDEN, self.NONLINEARITY)
	    return tf.reduce_sum(tf.add(x * tf.log(1e-10 + pi),  (1-x) * tf.log(1e-10 + 1 - pi)), axis=1)


    def _compute_logpy(self, y, x, z):
	""" compute the likelihood of every element in y under p(y|x,z) """
	h = tf.concat([x,z], axis=1)
	y_ = dgm._forward_pass_Cat_logits(h, self.Pzx_y, self.NUM_HIDDEN, self.NONLINEARITY)
	return -tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_)

    
    def _compute_loss_weights(self):
    	""" Compute scaling weights for the loss function """
        self.labeled_weight = tf.cast(tf.divide(self.N , tf.multiply(self.NUM_LABELED, self.LABELED_BATCH_SIZE)), tf.float32)
        self.unlabeled_weight = tf.cast(tf.divide(self.N , tf.multiply(self.NUM_UNLABELED, self.UNLABELED_BATCH_SIZE)), tf.float32)

    
    def compute_acc(self, x, y):
	y_, yq = self.predict(x)
	acc =  tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_,axis=1), tf.argmax(y, axis=1)), tf.float32))
	acc_q =  tf.reduce_mean(tf.cast(tf.equal(tf.argmax(yq,axis=1), tf.argmax(y, axis=1)), tf.float32))
	return acc, acc_q

    def _binarize(self, x):
	return np.random.binomial(1, x)


    def _process_data(self, data):
    	""" Extract relevant information from data_gen """
    	self.N = data.N
    	self.TRAINING_SIZE = data.TRAIN_SIZE   			 # training set size
	self.TEST_SIZE = data.TEST_SIZE                          # test set size
	self.NUM_LABELED = data.NUM_LABELED    			 # number of labeled instances
	self.NUM_UNLABELED = data.NUM_UNLABELED                  # number of unlabeled instances
	self.X_DIM = data.INPUT_DIM            			 # input dimension     
	self.NUM_CLASSES = data.NUM_CLASSES                      # number of classes
	self.alpha = self.alpha / self.NUM_LABELED               # weighting for additional term


    def _create_placeholders(self):
 	""" Create input/output placeholders """
	self.x_labeled = tf.placeholder(tf.float32, shape=[self.LABELED_BATCH_SIZE, self.X_DIM], name='labeled_input')
    	self.x_unlabeled = tf.placeholder(tf.float32, shape=[self.UNLABELED_BATCH_SIZE, self.X_DIM], name='unlabeled_input')
    	self.labels = tf.placeholder(tf.float32, shape=[self.LABELED_BATCH_SIZE, self.NUM_CLASSES], name='labels')
	self.x_train = tf.placeholder(tf.float32, shape=[self.TRAINING_SIZE, self.X_DIM], name='x_train')
	self.y_train = tf.placeholder(tf.float32, shape=[self.TRAINING_SIZE, self.NUM_CLASSES], name='y_train')
	self.x_test = tf.placeholder(tf.float32, shape=[self.TEST_SIZE, self.X_DIM], name='x_test')
	self.y_test = tf.placeholder(tf.float32, shape=[self.TEST_SIZE, self.NUM_CLASSES], name='y_test')
    	


    def _initialize_networks(self):
    	""" Initialize all model networks """
	if self.TYPE_PX == 'Gaussian':
      	    self.Pz_x = dgm._init_Gauss_net(self.Z_DIM, self.NUM_HIDDEN, self.X_DIM)
	elif self.TYPE_PX == 'Bernoulli':
	    self.Pz_x = dgm._init_Cat_net(self.Z_DIM, self.NUM_HIDDEN, self.X_DIM)
    	self.Pzx_y = dgm._init_Cat_net(self.Z_DIM+self.X_DIM, self.NUM_HIDDEN, self.NUM_CLASSES)
    	self.Qxy_z = dgm._init_Gauss_net(self.X_DIM+self.NUM_CLASSES, self.NUM_HIDDEN, self.Z_DIM)
    	self.Qx_y = dgm._init_Cat_net(self.X_DIM, self.NUM_HIDDEN, self.NUM_CLASSES)

    

    def _generate_class(self, k, num):
	""" create one-hot encoding of class k with length num """
	y = np.zeros(shape=(num, self.NUM_CLASSES))
	y[:,k] = 1
	return tf.constant(y, dtype=tf.float32)


    def _hook_loss(self, epoch, SKIP_STEP, total_loss, l_l, l_u, l_e):
    	print('Epoch {}: Total:{:5.1f}, Labeled:{:5.1f}, unlabeled:{:5.1f}, Additional:{:5.1f}'.format(epoch, 
											 	      total_loss/SKIP_STEP,
												      l_l/SKIP_STEP,
												      l_u/SKIP_STEP,
												      l_e/SKIP_STEP))

      
    def _allocate_directory(self):
	return 'graphs/gssl-'
