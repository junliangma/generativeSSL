from __future__ import absolute_import
from __future__ import division 
from __future__ import print_function

import sys, os, pdb

from models.model import model 

import numpy as np
import utils.dgm as dgm 

import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector


""" 
Generative models for labels with stochastic inputs: P(Z)P(X|Z)P(Y|X,Z,W)P(W) 
Here we implement Bayesian training for the model (hence Bgssl)

"""

class bgssl(model):
   
    def __init__(self, Z_DIM=2, LEARNING_RATE=0.005, NUM_HIDDEN=[4], ALPHA=0.1, TYPE_PX='Gaussian', NONLINEARITY=tf.nn.relu, initVar=-5, start_temp=0.8, BATCHNORM=False, 
                 LABELED_BATCH_SIZE=16, UNLABELED_BATCH_SIZE=128, NUM_EPOCHS=75, Z_SAMPLES=1, temperature_epochs=None, BINARIZE=False, verbose=1, logging=False, ckpt=None):
	
	super(bgssl, self).__init__(Z_DIM, LEARNING_RATE, NUM_HIDDEN, TYPE_PX, NONLINEARITY, BATCHNORM, temperature_epochs, start_temp, NUM_EPOCHS, Z_SAMPLES, BINARIZE, logging, ckpt=ckpt)

    	self.LABELED_BATCH_SIZE = LABELED_BATCH_SIZE         # labeled batch size 
	self.UNLABELED_BATCH_SIZE = UNLABELED_BATCH_SIZE     # labeled batch size 
    	self.alpha = ALPHA 				     # weighting for additional term
	self.initVar = initVar                               # initial variance for BNN prior weight distribution
    	self.verbose = verbose				     # control output: 0-ELBO, 1-accuracy, 2-Q-accuracy
	self.name = 'bgssl'    

    def fit(self, Data):
    	self._data_init(Data)
        ## define loss function
	self._compute_loss_weights()
        L_l = tf.reduce_mean(self._labeled_loss(self.x_labeled, self.labels))
        L_u = tf.reduce_mean(self._unlabeled_loss(self.x_unlabeled))
        L_e = tf.reduce_mean(self._qxy_loss(self.x_labeled, self.labels))
        self.loss = -(L_l + L_u + self.alpha*L_e)

        ## define optimizer
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss, global_step=self.global_step)
	
	## compute accuracies
	self.train_acc, train_acc_q= self.compute_acc(self.x_train, self.y_train)
	self.test_acc, test_acc_q = self.compute_acc(self.x_test, self.y_test)
	average_var = self._average_variance_W()
	self._create_summaries(L_l, L_u, L_e)
	
        ## initialize session and train
        max_acc, epoch, step = 0.0, 0, 0
	with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            total_loss, l_l, l_u, l_e = 0.0, 0.0, 0.0, 0.0
	    saver = tf.train.Saver()
	    if self.LOGGING:
                writer = tf.summary.FileWriter(self.LOGDIR, sess.graph)

            while epoch < self.NUM_EPOCHS:
		self.phase = True
                x_labeled, labels, x_unlabeled, _ = Data.next_batch(self.LABELED_BATCH_SIZE, self.UNLABELED_BATCH_SIZE)
	        if self.BINARIZE == True:
	            x_labeled, x_unlabeled = self._binarize(x_labeled), self._binarize(x_unlabeled)

       	        _, loss_batch, l_lb, l_ub, l_eb, avg_var = sess.run([self.optimizer, self.loss, L_l, L_u, L_e, average_var], 
            			     	     		               feed_dict={self.x_labeled: x_labeled, 
	                   		    	 		               self.labels: labels,
	          	            		     		               self.x_unlabeled: x_unlabeled,
	        						               self.beta:self.schedule[epoch]})

	        total_loss, l_l, l_u, l_e, step = total_loss+loss_batch, l_l+l_lb, l_u+l_ub, l_e+l_eb, step+1
                if Data._epochs_unlabeled > epoch:
	            fd = self._printing_feed_dict(Data, x_labeled, labels)
	            acc_train, acc_test, summary_acc = sess.run([self.train_acc, self.test_acc, self.summary_op_acc], fd)
	        
	            self._save_model(saver, sess, step, max_acc, acc_test)
	            max_acc = acc_test if acc_test > max_acc else max_acc
	            if self.LOGGING: 
	        	writer.add_summary(summary_acc, global_step=epoch)

	            if self.verbose == 0:
	        	""" Print ELBOs and accuracy """
	                self._print_verbose1(epoch, step, total_loss, l_l, l_u, l_e, acc_train, acc_test)
                        total_loss, l_l, l_u, l_e, step = 0.0, 0.0, 0.0, 0.0, 0
                    
	            elif self.verbose == 1:
	        	""" Print semi-supervised statistics"""
	        	self._print_verbose1(epoch, fd, sess, avg_var)        	    	

	            elif self.verbose == 2:
	                acc_train, acc_test,  = sess.run([train_acc_q, test_acc_q],
	        				         feed_dict = {self.x_train:x_train,
	        				     	              self.y_train:Data.data['y_train'],
	        						      self.x_test:x_test,
	        						      self.y_test:Data.data['y_test']})

	                print('At epoch {}: Training: {:5.3f}, Test: {:5.3f}'.format(epoch, acc_train, acc_test))
	            epoch += 1 
	    if self.LOGGING:
	        writer.close()
    
########### PREDICTION METHODS ############
    

    def predict_new(self, x, n_iters=20):
        saver = tf.train.Saver()
        with tf.Session() as session:
            ckpt = tf.train.get_checkpoint_state(self.ckpt_dir)
            saver.restore(session, ckpt.model_checkpoint_path)
	    self.phase = False
            preds = session.run([self.predict(x, n_iters)])
        return preds[0][0]


    def predict(self, x, n_w=10, n_iters=20):
	self._sample_W()
	y_, yq = self._predict_condition_W(x, n_iters)
	y_ = tf.expand_dims(y_, axis=2)
	for i in range(n_w-1):
   	    self._sample_W()
            y_new, _ = self._predict_condition_W(x, n_iters) 
	    y_ = tf.concat([y_, tf.expand_dims(y_new, axis=2)], axis=2)
        return tf.reduce_mean(y_, axis=2), yq

    def _predict_condition_W(self, x, n_iters=20):
	y_ = dgm._forward_pass_Cat(x, self.Qx_y, self.NUM_HIDDEN, self.NONLINEARITY, self.batchnorm, self.phase)
	yq = y_
	y_ = tf.one_hot(tf.argmax(y_, axis=1), self.NUM_CLASSES)
	y_samps = tf.expand_dims(y_, axis=2)
	for i in range(n_iters):
	    _, _, z = self._sample_Z(x, y_, self.Z_SAMPLES)
	    h = tf.concat([x, z], axis=1)
	    y_ = dgm._forward_pass_Cat_bnn(h, self.Wtilde, self.Pzx_y, self.NUM_HIDDEN, self.NONLINEARITY, self.batchnorm, self.phase)
	    #y_ = dgm._forward_pass_Cat(h, self.Wtilde, self.NUM_HIDDEN, self.NONLINEARITY, self.batchnorm, self.phase)
	    y_samps = tf.concat([y_samps, tf.expand_dims(y_, axis=2)], axis=2)
	    y_ = tf.one_hot(tf.argmax(y_, axis=1), self.NUM_CLASSES)
	return tf.reduce_mean(y_samps, axis=2), yq

    def sample_y(self, x, n_w=10, n_iters=20):
	self._sample_W()
	y_, yq = self._predict_condition_W(x, n_iters)
	y_ = tf.expand_dims(y_, axis=2)
	for i in range(n_w-1):
   	    self._sample_W()
            y_new, _ = self._predict_condition_W(x, n_iters) 
	    y_ = tf.concat([y_, tf.expand_dims(y_new, axis=2)], axis=2)
        return y_

###########################################



########### GENERATIVE METHODS ############


    def _sample_xy(self, n_samples=int(1e3)):
	saver = tf.train.Saver()
	with tf.Session() as session:
	    ckpt = tf.train.get_checkpoint_state(self.ckpt_dir)
	    saver.restore(session, ckpt.model_checkpoint_path)
	    self.phase = False
	    z_ = np.random.normal(size=(n_samples, self.Z_DIM)).astype('float32')
	    if self.TYPE_PX=='Gaussian':
                x_ = dgm._forward_pass_Gauss(z_, self.Pz_x, self.NUM_HIDDEN, self.NONLINEARITY, self.batchnorm, self.phase)
            else:
                x_ = dgm._forward_pass_Bernoulli(z_, self.Pz_x, self.NUM_HIDDEN, self.NONLINEARITY, self.batchnorm, self.phase)
            h = tf.concat([x_[0],z_], axis=1)
	    self._sample_W()
	    y_ = dgm._forward_pass_Cat_bnn(h, self.Wtilde, self.Pzx_y, self.NUM_HIDDEN, self.NONLINEARITY, self.batchnorm, self.phase)
	    ##y_ = dgm._forward_pass_Cat(h, self.Wtilde, self.NUM_HIDDEN, self.NONLINEARITY, self.batchnorm, self.phase)
            x,y = session.run([x_,y_])
        return x[0],y


    def _sample_Z(self, x, y, n_samples):
	""" Sample from Z with the reparamterization trick """
	h = tf.concat([x, y], axis=1)
	mean, log_var = dgm._forward_pass_Gauss(h, self.Qxy_z, self.NUM_HIDDEN, self.NONLINEARITY, self.batchnorm, self.phase)
	eps = tf.random_normal([tf.shape(x)[0], self.Z_DIM], dtype=tf.float32)
	return mean, log_var, mean + tf.sqrt(tf.exp(log_var)) * eps


    def _sample_W(self):
	""" Sample from W with the reparamterization trick """
	for i in range(len(self.NUM_HIDDEN)):
	    weight_name, bias_name = 'W'+str(i), 'b'+str(i)
	    mean_W, mean_b = self.Pzx_y['W'+str(i)+'_mean'], self.Pzx_y['b'+str(i)+'_mean']
            logvar_W, logvar_b = self.Pzx_y['W'+str(i)+'_logvar'], self.Pzx_y['b'+str(i)+'_logvar']
	    eps_W = tf.random_normal(mean_W.get_shape(), dtype=tf.float32)
	    eps_b = tf.random_normal(mean_b.get_shape(), dtype=tf.float32)
	    self.Wtilde[weight_name] = mean_W + tf.sqrt(tf.exp(logvar_W)) * eps_W
	    self.Wtilde[bias_name] = mean_b + tf.sqrt(tf.exp(logvar_b)) * eps_b
	mean_W, logvar_W = self.Pzx_y['Wout_mean'], self.Pzx_y['Wout_logvar']
	mean_b, logvar_b = self.Pzx_y['bout_mean'], self.Pzx_y['bout_logvar']
	eps_W = tf.random_normal(mean_W.get_shape(), dtype=tf.float32)
	eps_b = tf.random_normal(mean_b.get_shape(), dtype=tf.float32)
	self.Wtilde['Wout'] = mean_W + tf.sqrt(tf.exp(logvar_W)) * eps_W
	self.Wtilde['bout'] = mean_b + tf.sqrt(tf.exp(logvar_b)) * eps_b


###########################################



########### LOSS COMPUTATIONS #############

    def _labeled_loss_W(self, x, y):
	""" Compute necessary terms for labeled loss (per data point) """
	d = tf.cast(self.TRAINING_SIZE, tf.float32)
	q_mean, q_logvar, z = self._sample_Z(x, y, self.Z_SAMPLES)
	l_px = self._compute_logpx(x, z)
	l_py = self._compute_logpy(y, x, z)
	l_pz = dgm._gauss_logp(z, tf.zeros_like(z), tf.log(tf.ones_like(z)))
	l_qz = dgm._gauss_logp(z, q_mean, q_logvar)
	klz = dgm._gauss_kl(q_mean, q_logvar)
	klw = self._kl_W() / d
	return l_px + l_py  + self.beta * (l_pz - l_qz) - klw


    def _labeled_loss(self, x, y):
	self._sample_W()
	return self._labeled_loss_W(x,y)


    def _unlabeled_loss(self, x):
	""" Compute necessary terms for unlabeled loss (per data point) """
	weights = dgm._forward_pass_Cat(x, self.Qx_y, self.NUM_HIDDEN, self.NONLINEARITY, self.batchnorm, self.phase)
	self._sample_W()
	EL_l = 0 
	for i in range(self.NUM_CLASSES):
	    y = self._generate_class(i, x.get_shape()[0])
	    EL_l += tf.multiply(weights[:,i], self._labeled_loss_W(x,y))
	ent_qy = -tf.reduce_sum(weights * tf.log(1e-10 + weights), axis=1)
	return EL_l + ent_qy


    def _qxy_loss(self, x, y):
	y_ = dgm._forward_pass_Cat_logits(x, self.Qx_y, self.NUM_HIDDEN, self.NONLINEARITY, self.batchnorm, self.phase)
	return -tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_)


###########################################





############ HELPER FUNCTIONS #############

    def _compute_logpx(self, x, z):
	""" compute the likelihood of every element in x under p(x|z) """
	if self.TYPE_PX == 'Gaussian':
	    mean, log_var = dgm._forward_pass_Gauss(z,self.Pz_x, self.NUM_HIDDEN, self.NONLINEARITY, self.batchnorm, self.phase)
	    return dgm._gauss_logp(x, mean, log_var)
	elif self.TYPE_PX == 'Bernoulli':
	    pi = dgm._forward_pass_Bernoulli(z, self.Pz_x, self.NUM_HIDDEN, self.NONLINEARITY, self.batchnorm, self.phase)
	    return tf.reduce_sum(tf.add(x * tf.log(1e-10 + pi),  (1-x) * tf.log(1e-10 + 1 - pi)), axis=1)


    def _compute_logpy(self, y, x, z):
	""" compute the likelihood of every element in y under p(y|x,z, w) with sampled w"""
	h = tf.concat([x,z], axis=1)
	y_ = dgm._forward_pass_Cat_logits_bnn(h, self.Wtilde, self.Pzx_y, self.NUM_HIDDEN, self.NONLINEARITY, self.batchnorm, self.phase)
	#y_ = dgm._forward_pass_Cat_logits(h, self.Wtilde, self.NUM_HIDDEN, self.NONLINEARITY, self.batchnorm, self.phase)
	return -tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_)

    def _kl_W(self):
    	kl = 0
    	for i in range(len(self.NUM_HIDDEN)):
    	    mean, logvar = self.Pzx_y['W'+str(i)+'_mean'], self.Pzx_y['W'+str(i)+'_logvar']
    	    kl += tf.reduce_sum(dgm._gauss_kl(mean, logvar))
    	    mean, logvar = self.Pzx_y['b'+str(i)+'_mean'], self.Pzx_y['b'+str(i)+'_logvar']
    	    kl += tf.reduce_sum(dgm._gauss_kl(tf.expand_dims(mean,1), tf.expand_dims(logvar,1)))
    	mean, logvar = self.Pzx_y['Wout_mean'], self.Pzx_y['Wout_logvar']
    	kl += tf.reduce_sum(dgm._gauss_kl(mean, logvar))
    	mean, logvar = self.Pzx_y['bout_mean'], self.Pzx_y['bout_logvar']
    	kl += tf.reduce_sum(dgm._gauss_kl(tf.expand_dims(mean,1), tf.expand_dims(logvar,1)))
	return kl
   
    def _average_variance_W(self):
	total_var, num_params = 0,0
        for i in range(len(self.NUM_HIDDEN)):
            variances = tf.reshape(self.Pzx_y['W'+str(i)+'_logvar'], [-1])
            total_var += tf.reduce_sum(tf.exp(variances))
            num_params += tf.cast(tf.shape(variances)[0], dtype=tf.float32)
            variances = tf.reshape(self.Pzx_y['b'+str(i)+'_logvar'], [-1])
            total_var += tf.reduce_sum(tf.exp(variances))
            num_params += tf.cast(tf.shape(variances)[0], dtype=tf.float32)
        variances = tf.reshape(self.Pzx_y['Wout_logvar'], [-1])
        total_var += tf.reduce_sum(tf.exp(variances))
        num_params += tf.cast(tf.shape(variances)[0], tf.float32)
        variances = tf.reshape(self.Pzx_y['bout_logvar'], [-1])
        total_var += tf.reduce_sum(tf.exp(variances))
        num_params += tf.cast(tf.shape(variances)[0], dtype=tf.float32)
        return total_var / num_params
	

    def compute_acc(self, x, y):
	y_, yq = self.predict(x)
	acc =  tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_,axis=1), tf.argmax(y, axis=1)), tf.float32))
	acc_q =  tf.reduce_mean(tf.cast(tf.equal(tf.argmax(yq,axis=1), tf.argmax(y, axis=1)), tf.float32))
	return acc, acc_q



    def _initialize_networks(self):
    	""" Initialize all model networks """
	if self.TYPE_PX == 'Gaussian':
      	    self.Pz_x = dgm._init_Gauss_net(self.Z_DIM, self.NUM_HIDDEN, self.X_DIM, 'Pz_x_', bn=self.batchnorm)
	elif self.TYPE_PX == 'Bernoulli':
	    self.Pz_x = dgm._init_Cat_net(self.Z_DIM, self.NUM_HIDDEN, self.X_DIM, 'Pz_x_', bn=self.batchnorm)
    	self.Pzx_y  = dgm._init_Cat_bnn(self.Z_DIM+self.X_DIM, self.NUM_HIDDEN, self.NUM_CLASSES, 'Pzx_y', self.initVar)
	self.Wtilde = self._init_Wtilde(self.Z_DIM+self.X_DIM, self.NUM_HIDDEN, self.NUM_CLASSES, 'W_tilde_', bn=self.batchnorm)
    	self.Qxy_z  = dgm._init_Gauss_net(self.X_DIM+self.NUM_CLASSES, self.NUM_HIDDEN, self.Z_DIM, 'Qxy_z', bn=self.batchnorm)
    	self.Qx_y = dgm._init_Cat_net(self.X_DIM, self.NUM_HIDDEN, self.NUM_CLASSES, 'Qx_y', bn=self.batchnorm)


    def _generate_class(self, k, num):
	""" create one-hot encoding of class k with length num """
	y = np.zeros(shape=(num, self.NUM_CLASSES))
	y[:,k] = 1
	return tf.constant(y, dtype=tf.float32)


    def _init_Wtilde(self, n_in, architecture, n_out, vname, bn=False):
        """ Initialize the weights of a network with batch normalization parameterizeing a Categorical distribution """
        weights = {}
        if bn:
            for i, neurons in enumerate(architecture):
                scale, beta, mean, var = 'scale'+str(i), 'beta'+str(i), 'mean'+str(i), 'var'+str(i)
                weights[scale] = tf.Variable(tf.ones(architecture[i]), name=vname+scale)
                weights[beta] = tf.Variable(tf.zeros(architecture[i]), name=vname+beta)
                weights[mean] = tf.Variable(tf.zeros(architecture[i]), name=vname+mean, trainable=False)
                weights[var] = tf.Variable(tf.ones(architecture[i]), name=vname+var, trainable=False)
        return weights


########## ACQUISTION FUNCTIONS ###########

def _bald(self, x):
        pred_samples = self.sample_y(x)
        predictions = tf.reduce_mean(pred_samples, axis=2)
        H = -tf.reduce_sum(predictions * tf.log(1e-10+predictions), axis=1)
        E = tf.reduce_mean(-tf.reduce_sum(pred_samples * tf.log(1e-10+pred_samples), axis=1), axis=1)
        return H - E

###########################################

