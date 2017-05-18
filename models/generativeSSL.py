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
    def __init__(self, Z_DIM=2, LEARNING_RATE=0.005, NUM_HIDDEN=4, ALPHA=0.1, NONLINEARITY=tf.nn.relu,
		 LABELED_BATCH_SIZE=16, UNLABELED_BATCH_SIZE=128, NUM_EPOCHS=75, Z_SAMPLES=1, verbose=1):
    	## Step 1: define the placeholders for input and output
    	self.Z_DIM = Z_DIM                                   # stochastic inputs dimension       
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
    	
    	# Step 1: define the placeholders for input and output
    	self._create_placeholders()
    
       	## Step 2: define weights - setup all networks
        self._initialize_networks()
        
        ## Step 3: define the loss function
	self._compute_loss_weights()
        L_l = tf.reduce_sum(self._labeled_loss(self.x_labeled, self.labels))
        L_u = tf.reduce_sum(self._unlabeled_loss(self.x_unlabeled))
        L_e = self._qxy_loss(self.x_labeled, self.labels)
        self.loss = -tf.add_n([self.labeled_weight*L_l , self.unlabeled_weight*L_u , self.alpha*L_e], name='loss')
        
        ## Step 4: define optimizer
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
	
	## Step 5: compute accuracies
	train_acc, train_acc_q= self.compute_acc(self.x_train, self.y_train)
	test_acc, test_acc_q = self.compute_acc(self.x_test, self.y_test)
	
        ## Step 6: initialize session and train
        SKIP_STEP, epoch = 50, 0
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
                total_loss, l_l, l_u, l_e = total_loss+loss_batch, l_l+l_lb, l_u+l_ub, l_e+l_eb
                if Data._epochs_labeled > epoch:
		    epoch += 1
		    if self.verbose == 0:
		    	self._hook_loss(epoch, SKIP_STEP, total_loss, l_l, l_u, l_e)
        	        total_loss, l_l, l_u, l_e = 0.0, 0.0, 0.0, 0.0
        	    
		    elif self.verbose == 1:
		        acc_train, acc_test,  = sess.run([train_acc, test_acc],
						         feed_dict = {self.x_train:Data.data['x_train'],
						     	              self.y_train:Data.data['y_train'],
								      self.x_test:Data.data['x_test'],
								      self.y_test:Data.data['y_test']})
		        print('At epoch {}: Training: {:5.3f}, Test: {:5.3f}'.format(epoch, acc_train, acc_test))
        	    
		    elif self.verbose == 2:
		        acc_train, acc_test,  = sess.run([train_acci_q, test_acc_q],
						         feed_dict = {self.x_train:Data.data['x_train'],
						     	              self.y_train:Data.data['y_train'],
								      self.x_test:Data.data['x_test'],
								      self.y_test:Data.data['y_test']})
		        print('At epoch {}: Training: {:5.3f}, Test: {:5.3f}'.format(epoch, acc_train, acc_test))
	    writer.close()


    def predict(self, x, n_iters=10):
	y_ = self._forward_pass_Cat(x, self.Qx_y)
	yq = y_
	y_ = tf.one_hot(tf.argmax(y_, axis=1), self.NUM_CLASSES)
	y_samps = tf.expand_dims(y_, axis=2)
	for i in range(n_iters):
	    _, _, z = self._sample_Z(x, y_, self.Z_SAMPLES)
	    h = tf.concat([x, z], axis=1)
	    y_ = self._forward_pass_Cat(h, self.Pzx_y)
	    y_ = tf.one_hot(tf.argmax(y_, axis=1), self.NUM_CLASSES)
	    y_samps = tf.concat([y_samps, tf.expand_dims(y_, axis=2)], axis=2)
	return tf.reduce_mean(y_samps, axis=2), yq


    def _forward_pass_Gauss(self, x, weights):
	""" Forward pass through the network with weights as a dictionary """
	h = self.NONLINEARITY(tf.add(tf.matmul(x, weights['W_in']), weights['bias_in']))
	mean = tf.add(tf.matmul(h, weights['W_out_mean']), weights['bias_out_mean'])
	log_var = tf.nn.softplus(tf.add(tf.matmul(h, weights['W_out_var']), weights['bias_out_var']))
	return (mean, log_var)


    def _forward_pass_Cat(self, x, weights):
	""" Forward pass through the network with weights as a dictionary """
	return tf.nn.softmax(self._forward_pass_Cat_logits(x, weights))


    def _forward_pass_Cat_logits(self, x, weights):
	""" Forward pass through the network with weights as a dictionary """
	h = self.NONLINEARITY(tf.add(tf.matmul(x, weights['W_in']), weights['bias_in']))
	out = tf.add(tf.matmul(h, weights['W_out']), weights['bias_out'])
	return out


    def _sample_Z(self, x, y, n_samples):
	""" Sample from Z with the reparamterization trick """
	h = tf.concat([x, y], axis=1)
	mean, log_var = self._forward_pass_Gauss(h, self.Qxy_z)
	eps = tf.random_normal([n_samples, self.Z_DIM], 0, 1, dtype=tf.float32)
	return mean, log_var, tf.add(mean, tf.multiply(tf.sqrt(tf.exp(log_var)), eps))


    def _labeled_loss(self, x, y):
	""" Compute necessary terms for labeled loss (per data point) """
	q_mean, q_log_var, z = self._sample_Z(x, y, self.Z_SAMPLES)
	logpx = self._compute_logpx(x, z)
	logpy = self._compute_logpy(y, x, z)
	klz = self._gauss_kl(q_mean, tf.exp(q_log_var))
	return tf.add_n([logpx , logpy , -klz])


    def _unlabeled_loss(self, x):
	""" Compute necessary terms for unlabeled loss (per data point) """
	weights = self._forward_pass_Cat(x, self.Qx_y)
	EL_l = 0 
	for i in range(self.NUM_CLASSES):
	    y = self._generate_class(i, x.get_shape()[0])
	    EL_l += tf.multiply(weights[:,i], self._labeled_loss(x, y))
	ent_qy = -tf.reduce_sum(tf.multiply(weights, tf.log(weights)))
	return tf.add(EL_l, ent_qy)


    def _qxy_loss(self, x, y):
	y_ = self._forward_pass_Cat_logits(x, self.Qx_y)
	return -tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_))


    def _compute_logpx(self, x, z):
	""" compute the likelihood of every element in x under p(x|z) """
	mean, log_var = self._forward_pass_Gauss(z,self.Pz_x)
	mvn = tf.contrib.distributions.MultivariateNormalDiag(loc=mean, scale_diag=tf.exp(log_var))
	return mvn.log_prob(x)


    def _compute_logpy(self, y, x, z):
	""" compute the likelihood of every element in y under p(y|x,z) """
	h = tf.concat([x,z], axis=1)
	y_ = self._forward_pass_Cat_logits(h, self.Pzx_y)
	return -tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_)

    def _gauss_kl(self, mean, sigma):
	""" compute the KL-divergence of a Gaussian against N(0,1) """
	mean_0, sigma_0 = tf.zeros_like(mean), tf.ones_like(sigma)
	mvnQ = tf.contrib.distributions.MultivariateNormalDiag(loc=mean, scale_diag=sigma)
	prior = tf.contrib.distributions.MultivariateNormalDiag(loc=mean_0, scale_diag=sigma_0)
	return tf.contrib.distributions.kl(mvnQ, prior)

    
    def _compute_loss_weights(self):
    	""" Compute scaling weights for the loss function """
        self.labeled_weight = tf.cast(tf.divide(self.N , tf.multiply(self.NUM_LABELED, self.LABELED_BATCH_SIZE)), tf.float32)
        self.unlabeled_weight = tf.cast(tf.divide(self.N , tf.multiply(self.NUM_UNLABELED, self.UNLABELED_BATCH_SIZE)), tf.float32)



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
    


    def compute_acc(self, x, y):
	y_, yq = self.predict(x)
	acc =  tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_,axis=1), tf.argmax(y, axis=1)), tf.float32))
	acc_q =  tf.reduce_mean(tf.cast(tf.equal(tf.argmax(yq,axis=1), tf.argmax(y, axis=1)), tf.float32))
	return acc, acc_q


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
