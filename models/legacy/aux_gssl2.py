from __future__ import absolute_import
from __future__ import division 
from __future__ import print_function

from models.model import model

import sys, os, pdb

import numpy as np
import utils.dgm as dgm 

import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector


""" 
Generative models for labels with stochastic inputs and auxiliary variable: P(Z)P(X|Z)P(Y|X,Z), P(A|X,Y,Z) 
Inference network: q(a,z,y|x) = q(a|x) * q(y|a,x) * q(z|a,y,x) 
"""

class agssl(model):
   
    def __init__(self, Z_DIM=2, LEARNING_RATE=0.005, NUM_HIDDEN=[4], ALPHA=0.1, TYPE_PX='Gaussian', NONLINEARITY=tf.nn.relu, temperature_epochs=None, start_temp=0.5, l2_reg=0.0,
                 BATCHNORM=False, LABELED_BATCH_SIZE=16, UNLABELED_BATCH_SIZE=128, NUM_EPOCHS=75, eval_samps=None, Z_SAMPLES=1, BINARIZE=False, A_DIM=1,
		 verbose=1, logging=False, ckpt=None):
    	
	super(agssl, self).__init__(Z_DIM, LEARNING_RATE, NUM_HIDDEN, TYPE_PX, NONLINEARITY, BATCHNORM, temperature_epochs, start_temp, NUM_EPOCHS, Z_SAMPLES, BINARIZE, logging, eval_samps=eval_samps, ckpt=ckpt)

    	self.LABELED_BATCH_SIZE = LABELED_BATCH_SIZE         # labeled batch size 
	self.UNLABELED_BATCH_SIZE = UNLABELED_BATCH_SIZE     # labeled batch size 
	self.A_DIM = A_DIM                                   # auxiliary variable dimension
    	self.alpha = ALPHA 				     # weighting for additional term
    	self.verbose = verbose				     # control output: 0-ELBO, 1-accuracy, 2-Q-accuracy
        self.l2_reg = l2_reg                                 # factor for l2 regularization
	self.name = 'auxiliary_gssl'

    def fit(self, Data):
        self._data_init(Data)
	## define loss function
        L_l = tf.reduce_mean(self._labeled_loss(self.x_labeled, self.labels))
        L_u = tf.reduce_mean(self._unlabeled_loss(self.x_unlabeled))
        L_e = tf.reduce_mean(self._qxy_loss(self.x_labeled, self.labels))
	weight_prior = self._weight_regularization() / (self.LABELED_BATCH_SIZE+self.UNLABELED_BATCH_SIZE)
        self.loss = -(L_l + L_u + self.alpha*L_e - self.l2_reg * weight_prior)
        
        ## define optimizer 
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss, global_step=self.global_step)
	
	## compute accuracies
	self.train_acc, self.train_acc_q= self.compute_acc(self.x_train, self.y_train)
	self.test_acc, self.test_acc_q = self.compute_acc(self.x_test, self.y_test)
	self._create_summaries(L_l, L_u, L_e)

        ## initialize session and train
        max_acc, epoch, step = 0, 0, 0
	with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            total_loss, l_l, l_u, l_e = 0.0, 0.0, 0.0, 0.0
	    saver = tf.train.Saver()
	    #self.epoch_test_acc.append(sess.run(self.test_acc,
		#				{self.x_test:Data.data['x_test'],
		#				 self.y_test:Data.data['y_test']}))
	    if self.LOGGING:
                writer = tf.summary.FileWriter(self.LOGDIR, sess.graph)


            while epoch < self.NUM_EPOCHS:
		self.phase=True
                x_labeled, labels, x_unlabeled, _ = Data.next_batch(self.LABELED_BATCH_SIZE, self.UNLABELED_BATCH_SIZE)
		if self.BINARIZE == True:
	 	    x_labeled, x_unlabeled = self._binarize(x_labeled), self._binarize(x_unlabeled)
		
            	_, loss_batch, l_lb, l_ub, l_eb, summary_elbo = sess.run([self.optimizer, self.loss, L_l, L_u, L_e, self.summary_op_elbo], 
            			     	     		                  feed_dict={self.x_labeled: x_labeled, 
		           		    	 		           self.labels: labels,
		  	            		     		           self.x_unlabeled: x_unlabeled,
									   self.beta:self.schedule[epoch]})
		self.train_elbo.append(loss_batch)
		if self.LOGGING:
             	    writer.add_summary(summary_elbo, global_step=self.global_step)
		total_loss, l_l, l_u, l_e, step = total_loss+loss_batch, l_l+l_lb, l_u+l_ub, l_e+l_eb, step+1

                if Data._epochs_unlabeled > epoch:
		    epoch += 1
		    fd = self._printing_feed_dict(Data, x_labeled, labels)
		    acc_train, acc_test, summary_acc = sess.run([self.train_acc, self.test_acc, self.summary_op_acc], fd)
		    self.epoch_test_acc.append(acc_test)
        	    
		    self._save_model(saver,sess,step,max_acc,acc_test)
		    max_acc = acc_test if acc_test > max_acc else max_acc
		    if self.LOGGING:
         	        writer.add_summary(summary_acc, global_step=epoch)
		    
		    if self.verbose==0:
		        """ Print ELBOs and accuracy"""
		    	self._print_verbose0(epoch, step, total_loss, l_l, l_u, l_e, acc_train, acc_test)
        	        total_loss, l_l, l_u, l_e, step = 0.0, 0.0, 0.0, 0.0, 0
		    
		    elif self.verbose==1:
		        """ Print semi-supervised aspects """
			self._print_verbose1(epoch, fd, sess, acc_train, acc_test)
	
		    elif self.verbose==2:
		        acc_train, acc_test,  = sess.run([train_acc_q, test_acc_q], feed_dict=fd)
		        print('At epoch {}: Training: {:5.3f}, Test: {:5.3f}'.format(epoch, acc_train, acc_test))


   	    if self.LOGGING:
	        writer.close()


    def predict(self, x, n_iters=100):
	_, _, a = self._sample_a(x, self.Z_SAMPLES)
	h = tf.concat([x,a], axis=1)
	y_ = dgm._forward_pass_Cat(h, self.Qxa_y, self.NUM_HIDDEN, self.NONLINEARITY, self.batchnorm, self.phase)
	yq = y_
	y_ = tf.one_hot(tf.argmax(y_, axis=1), self.NUM_CLASSES)
	y_samps = tf.expand_dims(y_, axis=2)
	for i in range(n_iters):
	    _, _, z = self._sample_Z(x, y_, a, self.Z_SAMPLES)
	    h = tf.concat([x, z], axis=1)
	    y_ = dgm._forward_pass_Cat(h, self.Pzx_y, self.NUM_HIDDEN, self.NONLINEARITY, self.batchnorm, self.phase)
	    y_samps = tf.concat([y_samps, tf.expand_dims(y_, axis=2)], axis=2)
	    y_ = tf.one_hot(tf.argmax(y_, axis=1), self.NUM_CLASSES)
	return tf.reduce_mean(y_samps, axis=2), yq


    def encode(self, x, n_iters=100):
	_, _, a = self._sample_a(x, self.Z_SAMPLES)
	h = tf.concat([x,a], axis=1)
	y_ = dgm._forward_pass_Cat(h, self.Qxa_y, self.NUM_HIDDEN, self.NONLINEARITY, self.batchnorm, self.phase)
	y_ = tf.one_hot(tf.argmax(y_, axis=1), self.NUM_CLASSES) 
	_, _, z = self._sample_Z(x, y_, a, self.Z_SAMPLES)
	z_samps = tf.expand_dims(z, axis=2)
	for i in range(n_iters):
	    h = tf.concat([x, z], axis=1)
	    y_ = dgm._forward_pass_Cat(h, self.Pzx_y, self.NUM_HIDDEN, self.NONLINEARITY, self.batchnorm, self.phase)
	    y_ = tf.one_hot(tf.argmax(y_, axis=1), self.NUM_CLASSES)
	    _, _, z = self._sample_Z(x, y_, a, self.Z_SAMPLES)
	    z_samps = tf.concat([z_samps, tf.expand_dims(z, axis=2)], axis=2)
	return tf.reduce_mean(z_samps, axis=2)	


    def _sample_xy(self, n_samples=int(1e3)):
	saver = tf.train.Saver()
	with tf.Session() as session:
            ckpt = tf.train.get_checkpoint_state(self.ckpt_dir)
            saver.restore(session, ckpt.model_checkpoint_path)
	    self.phase=False
            z_ = np.random.normal(size=(n_samples, self.Z_DIM)).astype('float32')
            if self.TYPE_PX=='Gaussian':
                x_ = dgm._forward_pass_Gauss(z_, self.Pz_x, self.NUM_HIDDEN, self.NONLINEARITY, self.batchnorm, self.phase)[0]
            else:
            	x_ = dgm._forward_pass_Bernoulli(z_, self.Pz_x, self.NUM_HIDDEN, self.NONLINEARITY, self.batchnorm, self.phase)
            h = tf.concat([x_,z_], axis=1)
            y_ = dgm._forward_pass_Cat(h, self.Pzx_y, self.NUM_HIDDEN, self.NONLINEARITY, self.batchnorm, self.phase)
            x,y = session.run([x_,y_])
        return x,y

    def _sample_Z(self, x, y, a, n_samples):
	""" Sample from Z with the reparamterization trick """
	h = tf.concat([x, y, a], axis=1)
	mean, log_var = dgm._forward_pass_Gauss(h, self.Qxya_z, self.NUM_HIDDEN, self.NONLINEARITY, self.batchnorm, self.phase)
	eps = tf.random_normal([tf.shape(x)[0], self.Z_DIM], 0, 1, dtype=tf.float32)
	return mean, log_var, mean + tf.sqrt(tf.exp(log_var)) * eps

    def _sample_a(self, x, n_samples):
	""" Sample from a with the reparamterization trick """
	mean, log_var = dgm._forward_pass_Gauss(x, self.Qx_a, self.NUM_HIDDEN, self.NONLINEARITY, self.batchnorm, self.phase)
	eps = tf.random_normal([tf.shape(x)[0], self.A_DIM], 0, 1, dtype=tf.float32)
	return mean, log_var, mean + tf.sqrt(tf.exp(log_var)) * eps

    def _labeled_loss(self, x, y):
	""" Compute necessary terms for labeled loss (per data point) """
	qa_mean, qa_log_var, a = self._sample_a(x, self.Z_SAMPLES)
	q_mean, q_log_var, z = self._sample_Z(x, y, a, self.Z_SAMPLES)
	h = tf.concat([x,y,z], axis=1)
	pa_mean, pa_log_var = dgm._forward_pass_Gauss(h, self.Pzxy_a, self.NUM_HIDDEN, self.NONLINEARITY, self.batchnorm, self.phase)
	l_px = self._compute_logpx(x, z)
	l_py = self._compute_logpy(y, x, z)
	l_pz = dgm._gauss_logp(z, tf.zeros_like(z), tf.log(tf.ones_like(z)))
	l_pa = dgm._gauss_logp(a, pa_mean, pa_log_var)
	l_qz = dgm._gauss_logp(z, q_mean, q_log_var)
	l_qa = dgm._gauss_logp(a, qa_mean, qa_log_var)
	return l_px + l_py + self.beta * (l_pz + l_pa - l_qz - l_qa)


    def _unlabeled_loss(self, x):
	""" Compute necessary terms for unlabeled loss (per data point) """
	_, _, a = self._sample_a(x, self.Z_SAMPLES)
	h = tf.concat([x,a], axis=1)
	weights = dgm._forward_pass_Cat(h, self.Qxa_y, self.NUM_HIDDEN, self.NONLINEARITY, self.batchnorm, self.phase)
	EL_l = 0 
	for i in range(self.NUM_CLASSES):
	    y = self._generate_class(i, x.get_shape()[0])
	    EL_l += tf.multiply(weights[:,i], self._labeled_loss(x, y))
	ent_qy = -tf.reduce_sum(tf.multiply(weights, tf.log(1e-10 + weights)), axis=1)
	return EL_l + ent_qy, EL_l, ent_qy


    def _qxy_loss(self, x, y):
	_, _, a = self._sample_a(x, self.Z_SAMPLES)
	h = tf.concat([x,a], axis=1)
	y_ = dgm._forward_pass_Cat_logits(h, self.Qxa_y, self.NUM_HIDDEN, self.NONLINEARITY, self.batchnorm, self.phase)
	return -tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_)


    def _compute_logpy(self, y, x, z):
	""" compute the likelihood of every element in y under p(y|x,z) """
	h = tf.concat([x,z], axis=1)
	y_ = dgm._forward_pass_Cat_logits(h, self.Pzx_y, self.NUM_HIDDEN, self.NONLINEARITY, self.batchnorm, self.phase)
	return -tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_)


    def compute_acc(self, x, y):
	y_, yq = self.predict(x)
	acc =  tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_,axis=1), tf.argmax(y, axis=1)), tf.float32))
	acc_q =  tf.reduce_mean(tf.cast(tf.equal(tf.argmax(yq,axis=1), tf.argmax(y, axis=1)), tf.float32))
	return acc, acc_q


    def _initialize_networks(self):
    	""" Initialize all model networks """
	if self.TYPE_PX == 'Gaussian':
      	    self.Pz_x = dgm._init_Gauss_net(self.Z_DIM, self.NUM_HIDDEN, self.X_DIM, 'Pz_x_', self.batchnorm)
	elif self.TYPE_PX == 'Bernoulli':
	    self.Pz_x = dgm._init_Cat_net(self.Z_DIM, self.NUM_HIDDEN, self.X_DIM, 'Pz_x_', self.batchnorm)
    	self.Pzx_y = dgm._init_Cat_net(self.Z_DIM+self.X_DIM, self.NUM_HIDDEN, self.NUM_CLASSES, 'Pzx_y_', self.batchnorm)
	self.Pzxy_a = dgm._init_Gauss_net(self.Z_DIM+self.X_DIM+self.NUM_CLASSES, self.NUM_HIDDEN, self.A_DIM, 'P_A_', self.batchnorm) 
    	self.Qxya_z = dgm._init_Gauss_net(self.X_DIM+self.A_DIM+self.NUM_CLASSES, self.NUM_HIDDEN, self.Z_DIM, 'Qxya_z_', self.batchnorm)
    	self.Qx_a = dgm._init_Gauss_net(self.X_DIM, self.NUM_HIDDEN, self.A_DIM, 'Qx_a_', self.batchnorm)
    	self.Qxa_y = dgm._init_Cat_net(self.X_DIM+self.A_DIM, self.NUM_HIDDEN, self.NUM_CLASSES, 'Qxa_y_', self.batchnorm)

    

    def _generate_class(self, k, num):
	""" create one-hot encoding of class k with length num """
	y = np.zeros(shape=(num, self.NUM_CLASSES))
	y[:,k] = 1
	return tf.constant(y, dtype=tf.float32)


    def _print_verbose1(self,epoch, fd, sess, acc_train, acc_test, avg_var=None):
        am_test, alv_test, a_test = self._sample_a(self.x_test, 1)
        am_train, alv_train, a_train = self._sample_a(self.x_train, 1)
        zm_test, zlv_test, z_test = self._sample_Z(self.x_test,self.y_test, a_test, 1)
        zm_train, zlv_train, z_train = self._sample_Z(self.x_train,self.y_train, a_train, 1)
        lpx_test, lpx_train, klz_test, klz_train = sess.run([self._compute_logpx(self.x_test, z_test),
                                                                  self._compute_logpx(self.x_train, z_train),
                                                                  dgm._gauss_kl(zm_test, tf.exp(zlv_test)),
                                                                  dgm._gauss_kl(zm_train, tf.exp(zlv_train))], feed_dict=fd)

	print('Epoch: {}, logpx: {:5.3f}, klz: {:5.3f}, Train: {:5.3f}, Test: {:5.3f}'.format(epoch, np.mean(lpx_train), np.mean(klz_train), acc_train, acc_test))
