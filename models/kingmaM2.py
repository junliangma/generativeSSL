from __future__ import absolute_import
from __future__ import division 
from __future__ import print_function

import sys, os
parent_dir = os.getcwd()
path = os.path.dirname(parent_dir)
sys.path.append(path)

import numpy as np

import utils.dgm as dgm
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

import pdb

""" Implementation of Kingma et al (2014), M2:  P(Z)P(Y)P(X|Y,Z) """

class M2:
   
    def __init__(self, Z_DIM=2, LEARNING_RATE=0.005, NUM_HIDDEN=4, ALPHA=0.1, NONLINEARITY=tf.nn.relu, TYPE_PX='Gaussian',
		 LABELED_BATCH_SIZE=16, UNLABELED_BATCH_SIZE=128, NUM_EPOCHS=75, Z_SAMPLES=1, verbose=1):
    	## Step 1: define the placeholders for input and output
    	self.Z_DIM = Z_DIM                                   # stochastic inputs dimension       
    	self.TYPE_PX = TYPE_PX				     # Distribution over inputs X
    	self.NUM_HIDDEN = NUM_HIDDEN                         # number of hidden layers per network
    	self.NONLINEARITY = NONLINEARITY		     # activation functions	
    	self.lr = LEARNING_RATE 			     # learning rate
    	self.LABELED_BATCH_SIZE = LABELED_BATCH_SIZE         # labeled batch size 
	self.UNLABELED_BATCH_SIZE = UNLABELED_BATCH_SIZE     # labeled batch size 
    	self.alpha = ALPHA 				     # weighting for additional term
    	self.Z_SAMPLES = Z_SAMPLES 			     # number of monte-carlo samples
    	self.NUM_EPOCHS = NUM_EPOCHS                         # training epochs
    	self.LOGDIR = self._allocate_directory()             # logging directory
    	self.verbose = verbose				     # control output: 1 for ELBO, else accuracy
    

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
	train_acc = self.compute_acc(self.x_train, self.y_train)
	test_acc = self.compute_acc(self.x_test, self.y_test)
	
	
        ## initialize session and train
        SKIP_STEP, epoch, step = 50, 0, 0
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            total_loss, l_l, l_u, l_e = 0.0, 0.0, 0.0, 0.0
            writer = tf.summary.FileWriter(self.LOGDIR, sess.graph)
            while epoch < self.NUM_EPOCHS:
                batch = Data.next_batch(self.LABELED_BATCH_SIZE, self.UNLABELED_BATCH_SIZE)
            	_, loss_batch, l_lb, l_ub, l_eb = sess.run([self.optimizer, self.loss, L_l, L_u, L_e], 
            			     	     		    feed_dict={self.x_labeled: batch[0], 
		           		    	 		       self.labels: batch[1],
		  	            		     		       self.x_unlabeled: batch[2]})
          
                total_loss, l_l, l_u, l_e, step = total_loss+loss_batch, l_l+l_lb, l_u+l_ub, l_e+l_eb, step+1
                if Data._epochs_labeled > epoch:
		    epoch += 1
		    if self.verbose == 0:
		    	self._hook_loss(epoch, step, total_loss, l_l, l_u, l_e)
        	        total_loss, l_l, l_u, l_e, step = 0.0, 0.0, 0.0, 0.0, 0
        	    
		    elif self.verbose == 1:
		        acc_train, acc_test,  = sess.run([train_acc, test_acc],
						         feed_dict = {self.x_train:Data.data['x_train'],
						     	              self.y_train:Data.data['y_train'],
								      self.x_test:Data.data['x_test'],
								      self.y_test:Data.data['y_test']})
		        print('At epoch {}: Training: {:5.3f}, Test: {:5.3f}'.format(epoch, acc_train, acc_test))
            	    
	    writer.close()


    def predict(self, x):
	return dgm._forward_pass_Cat(x, self.Qx_y, self.NUM_HIDDEN, self.NONLINEARITY)


    def _sample_Z(self, x, y, n_samples):
	""" Sample from Z with the reparamterization trick """
	h = tf.concat([x, y], axis=1)
	mean, log_var = dgm._forward_pass_Gauss(h, self.Qxy_z, self.NUM_HIDDEN, self.NONLINEARITY)
	eps = tf.random_normal([n_samples, self.Z_DIM], 0, 1, dtype=tf.float32)
	return mean, log_var, tf.add(mean, tf.multiply(tf.sqrt(tf.exp(log_var)), eps))


    def _labeled_loss(self, x, y):
	""" Compute necessary terms for labeled loss (per data point) """
	z_mean, z_log_var, z  = self._sample_Z(x, y, self.Z_SAMPLES)
	logpx = self._compute_logpx(x, z, y)
	logpy = self._compute_logpy(y)       
	KLz = dgm._gauss_kl(z_mean, tf.exp(z_log_var))
	return tf.add_n([logpx, logpy, -KLz], name='labeled_loss')

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


    def _compute_logpx(self, x, z, y):
	""" compute the likelihood of every element in x under p(x|z) """
	h = tf.concat([z,y], axis=1)
	if self.TYPE_PX == 'Gaussian':
	    mean, log_var = dgm._forward_pass_Gauss(h, self.Pzy_x, self.NUM_HIDDEN, self.NONLINEARITY)
	    mvn = tf.contrib.distributions.MultivariateNormalDiag(loc=mean, scale_diag=tf.exp(log_var))
	    return mvn.log_prob(x)
	elif self.TYPE_PX == 'Bernoulli':
	    pi = dgm._forward_pass_Bernoulli(h, self.Pzy_x, self.NUM_HIDDEN, self.NONLINEARITY)
	    return tf.reduce_sum(tf.add(x * tf.log(1e-10 + pi), (1-x) * tf.log(1e-10 + 1 - pi)), axis=1)


    def _compute_logpy(self, y):
	""" compute the likelihood of every element in y under p(y) """
	return -tf.reduce_mean(tf.multiply(y, tf.log(self.Py)), axis=1)


    def _compute_loss_weights(self):
    	""" Compute scaling weights for the loss function """
        self.labeled_weight = tf.cast(tf.divide(self.N , tf.multiply(self.NUM_LABELED, self.LABELED_BATCH_SIZE)), tf.float32)
        self.unlabeled_weight = tf.cast(tf.divide(self.N , tf.multiply(self.NUM_UNLABELED, self.UNLABELED_BATCH_SIZE)), tf.float32)

   
    def compute_acc(self, x, y):
	y_ = self.predict(x)
	return  tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_,axis=1), tf.argmax(y, axis=1)), tf.float32))


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
    	    self.Pzy_x = dgm._init_Gauss_net(self.Z_DIM + self.NUM_CLASSES, self.NUM_HIDDEN, self.X_DIM)
    	elif self.TYPE_PX =='Bernoulli':
    	    self.Pzy_x = dgm._init_Cat_net(self.Z_DIM + self.NUM_CLASSES, self.NUM_HIDDEN, self.X_DIM)	
    	self.Py = tf.constant((1./self.NUM_CLASSES)*np.ones(shape=(self.NUM_CLASSES,)), dtype=tf.float32)
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
	return 'graphs/default/'
