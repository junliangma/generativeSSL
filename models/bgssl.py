from __future__ import absolute_import
from __future__ import division 
from __future__ import print_function

import sys, os, pdb

import numpy as np
import utils.dgm as dgm 

import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector


""" 
Generative models for labels with stochastic inputs: P(Z)P(X|Z)P(Y|X,Z,W)P(W) 
Here we implement Bayesian training for the model (hence Bgssl)

"""

class bgssl:
   
    def __init__(self, Z_DIM=2, LEARNING_RATE=0.005, NUM_HIDDEN=[4], ALPHA=0.1, TYPE_PX='Gaussian', NONLINEARITY=tf.nn.relu, initVar=-5, 
                 LABELED_BATCH_SIZE=16, UNLABELED_BATCH_SIZE=128, NUM_EPOCHS=75, Z_SAMPLES=1, temperature_epochs=None, BINARIZE=False, verbose=1, logging=False):
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
	self.temperature_epochs = temperature_epochs         # number of epochs untill kl_W is fully on
	self.initVar = initVar                               # initial variance for BNN prior weight distribution
	self.BINARIZE = BINARIZE                             # sample inputs from Bernoulli distribution if true 
    	self.verbose = verbose				     # control output: 0-ELBO, 1-accuracy, 2-Q-accuracy
	self.logging = logging                               # whether or not to log
    

    def fit(self, Data):
    	self._process_data(Data)
    	
	self._create_placeholders() 
	self._set_schedule()
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
	average_var = self._average_variance_W()
	
        ## initialize session and train
        SKIP_STEP, epoch, step = 50, 0, 0
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            total_loss, l_l, l_u, l_e = 0.0, 0.0, 0.0, 0.0
	    saver = tf.train.Saver()
	    if self.logging:
                writer = tf.summary.FileWriter(self.LOGDIR, sess.graph)
            while epoch < self.NUM_EPOCHS:
		self.beta = self.schedule[epoch]
                x_labeled, labels, x_unlabeled, _ = Data.next_batch(self.LABELED_BATCH_SIZE, self.UNLABELED_BATCH_SIZE)
		if self.BINARIZE == True:
	 	    x_labeled, x_unlabeled = self._binarize(x_labeled), self._binarize(x_unlabeled)
            	_, loss_batch, l_lb, l_ub, l_eb, avg_var = sess.run([self.optimizer, self.loss, L_l, L_u, L_e, average_var], 
            			     	     		               feed_dict={self.x_labeled: x_labeled, 
		           		    	 		       self.labels: labels,
		  	            		     		       self.x_unlabeled: x_unlabeled})
                
		total_loss, l_l, l_u, l_e, step = total_loss+loss_batch, l_l+l_lb, l_u+l_ub, l_e+l_eb, step+1
                if Data._epochs_unlabeled > epoch:
		    saver.save(sess, self.ckpt_dir, global_step=step+1)
		    if self.verbose == 0:
		    	self._hook_loss(epoch, step, total_loss, l_l, l_u, l_e)
        	        total_loss, l_l, l_u, l_e, step = 0.0, 0.0, 0.0, 0.0, 0
        	    
		    elif self.verbose == 1:
			x_train = Data.data['x_train']
			x_test = Data.data['x_test']
			    
		        acc_train, acc_test  = sess.run([train_acc, test_acc],
						         feed_dict = {self.x_train:x_train,
						     	              self.y_train:Data.data['y_train'],
								      self.x_test:x_test,
								      self.y_test:Data.data['y_test']})
		        print('At epoch {}: Training: {:5.3f}, Test: {:5.3f}, Variance: {:5.5f}'.format(epoch, acc_train, acc_test, avg_var))
        	    
		    elif self.verbose == 2:
		        acc_train, acc_test,  = sess.run([train_acc_q, test_acc_q],
						         feed_dict = {self.x_train:x_train,
						     	              self.y_train:Data.data['y_train'],
								      self.x_test:x_test,
								      self.y_test:Data.data['y_test']})
		        print('At epoch {}: Training: {:5.3f}, Test: {:5.3f}'.format(epoch, acc_train, acc_test))
		    epoch += 1 
	    if self.logging:
	        writer.close()
    
    
    def predict_new(self, x, n_iters=100):
        predictions = self.predict(x, n_iters)
        saver = tf.train.Saver()
        with tf.Session() as session:
            ckpt = tf.train.get_checkpoint_state(self.ckpt_dir)
            saver.restore(session, ckpt.model_checkpoint_path)
            preds = session.run([predictions])
        return preds[0][0]


    def predict(self, x, n_w=10, n_iters=100):
	w_ = self._sample_W()
	y_, yq = self._predict_condition_W(x, w_)
	y_ = tf.expand_dims(y_, axis=2)
	for i in range(n_w-1):
   	    w_ = self._sample_W()
            y_new, _ = self._predict_condition_W(x, w_) 
	    y_ = tf.concat([y_, tf.expand_dims(y_new, axis=2)], axis=2)
        return tf.reduce_mean(y_, axis=2), yq

    def _predict_condition_W(self, x, w, n_iters=100):
	y_ = dgm._forward_pass_Cat(x, self.Qx_y, self.NUM_HIDDEN, self.NONLINEARITY)
	yq = y_
	y_ = tf.one_hot(tf.argmax(y_, axis=1), self.NUM_CLASSES)
	y_samps = tf.expand_dims(y_, axis=2)
	for i in range(n_iters):
	    _, _, z = self._sample_Z(x, y_, self.Z_SAMPLES)
	    h = tf.concat([x, z], axis=1)
	    y_ = dgm._forward_pass_Cat(h, w, self.NUM_HIDDEN, self.NONLINEARITY)
	    y_samps = tf.concat([y_samps, tf.expand_dims(y_, axis=2)], axis=2)
	    y_ = tf.one_hot(tf.argmax(y_, axis=1), self.NUM_CLASSES)
	return tf.reduce_mean(y_samps, axis=2), yq



    def _sample_Z(self, x, y, n_samples):
	""" Sample from Z with the reparamterization trick """
	h = tf.concat([x, y], axis=1)
	mean, log_var = dgm._forward_pass_Gauss(h, self.Qxy_z, self.NUM_HIDDEN, self.NONLINEARITY)
	eps = tf.random_normal([tf.shape(x)[0], self.Z_DIM], dtype=tf.float32)
	return mean, log_var, tf.add(mean, tf.multiply(tf.nn.softplus(log_var), eps))


    def _sample_W(self):
	""" Sample from W with the reparamterization trick """
	weights = {}
	for i in range(len(self.NUM_HIDDEN)):
	    weight_name, bias_name = 'W'+str(i), 'b'+str(i)
	    mean_W, mean_b = self.Pzx_y['W'+str(i)+'_mean'], self.Pzx_y['b'+str(i)+'_mean']
            logvar_W, logvar_b = self.Pzx_y['W'+str(i)+'_logvar'], self.Pzx_y['b'+str(i)+'_logvar']
	    eps_W = tf.random_normal(mean_W.get_shape(), dtype=tf.float32)
	    eps_b = tf.random_normal(mean_b.get_shape(), dtype=tf.float32)
	    weights[weight_name] = tf.add(mean_W, tf.multiply(tf.nn.softplus(logvar_W), eps_W))
	    weights[bias_name] = tf.add(mean_b, tf.multiply(tf.nn.softplus(logvar_b), eps_b))
	mean_W, logvar_W = self.Pzx_y['Wout_mean'], self.Pzx_y['Wout_logvar']
	mean_b, logvar_b = self.Pzx_y['bout_mean'], self.Pzx_y['bout_logvar']
	eps_W = tf.random_normal(mean_W.get_shape(), dtype=tf.float32)
	eps_b = tf.random_normal(mean_b.get_shape(), dtype=tf.float32)
	weights['Wout'] = tf.add(mean_W, tf.multiply(tf.nn.softplus(logvar_W), eps_W))
	weights['bout'] = tf.add(mean_b, tf.multiply(tf.nn.softplus(logvar_b), eps_b))
	return weights



    def _labeled_loss_W(self, x, y, w):
	""" Compute necessary terms for labeled loss (per data point) """
	q_mean, q_logvar, z = self._sample_Z(x, y, self.Z_SAMPLES)
	logpx = self._compute_logpx(x, z)
	logpy = self._compute_logpy(y, x, z, w)
	klz = dgm._gauss_kl(q_mean, tf.nn.softplus(q_logvar))
	klw = (self._kl_W() / tf.cast(self.N, tf.float32))
	return logpx + logpy  - klz - klw

    def _labeled_loss(self, x, y):
	w = self._sample_W()
	return self._labeled_loss_W(x,y,w)


    def _unlabeled_loss(self, x):
	""" Compute necessary terms for unlabeled loss (per data point) """
	weights = dgm._forward_pass_Cat(x, self.Qx_y, self.NUM_HIDDEN, self.NONLINEARITY)
	w = self._sample_W()
	EL_l = 0 
	for i in range(self.NUM_CLASSES):
	    y = self._generate_class(i, x.get_shape()[0])
	    EL_l += tf.multiply(weights[:,i], self._labeled_loss_W(x,y,w))
	ent_qy = -tf.reduce_sum(tf.multiply(weights, tf.log(1e-10 + weights)), axis=1)
	return tf.add(EL_l, ent_qy)


    def _qxy_loss(self, x, y):
	y_ = dgm._forward_pass_Cat_logits(x, self.Qx_y, self.NUM_HIDDEN, self.NONLINEARITY)
	return -tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_))


    def _compute_logpx(self, x, z):
	""" compute the likelihood of every element in x under p(x|z) """
	if self.TYPE_PX == 'Gaussian':
	    mean, log_var = dgm._forward_pass_Gauss(z,self.Pz_x, self.NUM_HIDDEN, self.NONLINEARITY)
	    mvn = tf.contrib.distributions.MultivariateNormalDiag(loc=mean, scale_diag=tf.nn.softplus(log_var))
	    return mvn.log_prob(x)
	elif self.TYPE_PX == 'Bernoulli':
	    pi = dgm._forward_pass_Bernoulli(z, self.Pz_x, self.NUM_HIDDEN, self.NONLINEARITY)
	    return tf.reduce_sum(tf.add(x * tf.log(1e-10 + pi),  (1-x) * tf.log(1e-10 + 1 - pi)), axis=1)


    def _compute_logpy(self, y, x, z, w):
	""" compute the likelihood of every element in y under p(y|x,z, w) with sampled w"""
	h = tf.concat([x,z], axis=1)
	y_ = dgm._forward_pass_Cat_logits(h, w, self.NUM_HIDDEN, self.NONLINEARITY)
	return -tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_)

    def _kl_W(self):
    	kl = 0
    	for i in range(len(self.NUM_HIDDEN)):
    	    mean, logvar = self.Pzx_y['W'+str(i)+'_mean'], self.Pzx_y['W'+str(i)+'_logvar']
    	    kl += dgm._gauss_kl(tf.reshape(mean, shape=[-1]), tf.reshape(tf.nn.softplus(logvar), shape=[-1]))
    	    mean, logvar = self.Pzx_y['b'+str(i)+'_mean'], self.Pzx_y['b'+str(i)+'_logvar']
    	    kl += dgm._gauss_kl(mean, tf.nn.softplus(logvar))
    	mean, logvar = self.Pzx_y['Wout_mean'], self.Pzx_y['Wout_logvar']
    	kl += dgm._gauss_kl(tf.reshape(mean, shape=[-1]), tf.reshape(tf.nn.softplus(logvar), shape=[-1]))
    	mean, logvar = self.Pzx_y['bout_mean'], self.Pzx_y['bout_logvar']
    	kl += dgm._gauss_kl(mean, tf.nn.softplus(logvar))
	return kl
   
    def _average_variance_W(self):
	total_var, num_params = 0,0
        for i in range(len(self.NUM_HIDDEN)):
            variances = tf.reshape(self.Pzx_y['W'+str(i)+'_logvar'], [-1])
            total_var += tf.reduce_sum(tf.nn.softplus(variances))
            num_params += tf.cast(tf.shape(variances)[0], dtype=tf.float32)
            variances = tf.reshape(self.Pzx_y['b'+str(i)+'_logvar'], [-1])
            total_var += tf.reduce_sum(tf.nn.softplus(variances))
            num_params += tf.cast(tf.shape(variances)[0], dtype=tf.float32)
        variances = tf.reshape(self.Pzx_y['Wout_logvar'], [-1])
        total_var += tf.reduce_sum(tf.nn.softplus(variances))
        num_params += tf.cast(tf.shape(variances)[0], tf.float32)
        variances = tf.reshape(self.Pzx_y['bout_logvar'], [-1])
        total_var += tf.reduce_sum(tf.nn.softplus(variances))
        num_params += tf.cast(tf.shape(variances)[0], dtype=tf.float32)
        return total_var / num_params
	
 
    def _compute_loss_weights(self):
    	""" Compute scaling weights for the loss function """
        #self.labeled_weight = tf.cast(tf.divide(self.N , tf.multiply(self.NUM_LABELED, self.LABELED_BATCH_SIZE)), tf.float32)
        #self.unlabeled_weight = tf.cast(tf.divide(self.N , tf.multiply(self.NUM_UNLABELED, self.UNLABELED_BATCH_SIZE)), tf.float32)
	self.labeled_weight = tf.cast(self.TRAINING_SIZE / self.LABELED_BATCH_SIZE, tf.float32)
	self.unlabeled_weight = tf.cast(self.TRAINING_SIZE / self.UNLABELED_BATCH_SIZE, tf.float32)
   
 
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
    	self.TRAINING_SIZE = data.TRAIN_SIZE   		 # training set size
	self.TEST_SIZE = data.TEST_SIZE                  # test set size
	self.NUM_LABELED = data.NUM_LABELED    		 # number of labeled instances
	self.NUM_UNLABELED = data.NUM_UNLABELED          # number of unlabeled instances
	self.X_DIM = data.INPUT_DIM            		 # input dimension     
	self.NUM_CLASSES = data.NUM_CLASSES              # number of classes
	self.alpha = self.alpha / self.NUM_LABELED       # weighting for additional term
	self.data_name = data.NAME                       # name of the dataset


    def _create_placeholders(self):
 	""" Create input/output placeholders """
	self.x_labeled = tf.placeholder(tf.float32, shape=[self.LABELED_BATCH_SIZE, self.X_DIM], name='labeled_input')
    	self.x_unlabeled = tf.placeholder(tf.float32, shape=[self.UNLABELED_BATCH_SIZE, self.X_DIM], name='unlabeled_input')
    	self.labels = tf.placeholder(tf.float32, shape=[self.LABELED_BATCH_SIZE, self.NUM_CLASSES], name='labels')
	self.x_train = tf.placeholder(tf.float32, shape=[self.TRAINING_SIZE, self.X_DIM], name='x_train')
	self.y_train = tf.placeholder(tf.float32, shape=[self.TRAINING_SIZE, self.NUM_CLASSES], name='y_train')
	self.x_test = tf.placeholder(tf.float32, shape=[self.TEST_SIZE, self.X_DIM], name='x_test')
	self.y_test = tf.placeholder(tf.float32, shape=[self.TEST_SIZE, self.NUM_CLASSES], name='y_test')
    	self._allocate_directory()            
    	


    def _initialize_networks(self):
    	""" Initialize all model networks """
	if self.TYPE_PX == 'Gaussian':
      	    self.Pz_x = dgm._init_Gauss_net(self.Z_DIM, self.NUM_HIDDEN, self.X_DIM, 'Pz_x_')
	elif self.TYPE_PX == 'Bernoulli':
	    self.Pz_x = dgm._init_Cat_net(self.Z_DIM, self.NUM_HIDDEN, self.X_DIM, 'Pz_x_')
    	self.Pzx_y = dgm._init_Cat_bnn(self.Z_DIM+self.X_DIM, self.NUM_HIDDEN, self.NUM_CLASSES, 'Pzx_y', self.initVar)
    	self.Qxy_z = dgm._init_Gauss_net(self.X_DIM+self.NUM_CLASSES, self.NUM_HIDDEN, self.Z_DIM, 'Qxy_z')
    	self.Qx_y = dgm._init_Cat_net(self.X_DIM, self.NUM_HIDDEN, self.NUM_CLASSES, 'Qx_y')


    def _set_schedule(self):
	if not self.temperature_epochs:
	    self.schedule = np.ones((self.NUM_EPOCHS,1))
	else:
	    warmup = np.expand_dims(np.arange(0, 1, 1./self.temperature_epochs),1)
	    plateau = np.ones(shape=(self.NUM_EPOCHS - self.temperature_epochs,1))
	    self.schedule = np.ravel(np.vstack((warmup, plateau)))
	self.beta = self.schedule[0]

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
	self.LOGDIR = 'graphs/gssl-'+self.data_name+'-'+str(self.lr)+'-'+str(self.NUM_LABELED)+'/'
        self.ckpt_dir = './ckpt/gssl-'+self.data_name+'-'+str(self.lr)+'-'+str(self.NUM_LABELED) + '/'
        if not os.path.isdir(self.ckpt_dir):
            os.mkdir(self.ckpt_dir)	
