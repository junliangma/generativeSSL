from __future__ import absolute_import
from __future__ import division 
from __future__ import print_function

from models.model import model

import sys, os, pdb

import numpy as np
import utils.dgm as dgm 

import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector


""" Generative models for labels with stochastic inputs: P(Z)P(X|Z)P(Y|X,Z) """

class igssl(model):
   
    def __init__(self, Z_DIM=2, LEARNING_RATE=0.005, NUM_HIDDEN=[4], ALPHA=0.1, TYPE_PX='Gaussian', NONLINEARITY=tf.nn.relu, temperature_epochs=None, start_temp=None, 
                 WARMUP=20, LABELED_BATCH_SIZE=16, UNLABELED_BATCH_SIZE=128, NUM_EPOCHS=75, Z_SAMPLES=1, BINARIZE=False, verbose=1, logging=True):
	
	super(igssl, self).__init__(Z_DIM, LEARNING_RATE, NUM_HIDDEN, TYPE_PX, NONLINEARITY, temperature_epochs, start_temp, NUM_EPOCHS, Z_SAMPLES, BINARIZE, logging)

    	self.LABELED_BATCH_SIZE = LABELED_BATCH_SIZE         # labeled batch size 
	self.UNLABELED_BATCH_SIZE = UNLABELED_BATCH_SIZE     # labeled batch size 
	self.WARMUP = WARMUP                                 # warmup period for VAE
    	self.alpha = ALPHA 				     # weighting for additional term
    	self.verbose = verbose				     # control output: 0-ELBO, 1-accuracy, 2-Q-accuracy
	self.name = 'igssl'                                  # model name
    

    def fit(self, Data):
    	self._process_data(Data)
    	
	self._create_placeholders() 
        self._initialize_networks()
	self._set_schedule()
        
	## define loss function
	self._compute_loss_weights()
        L_l = tf.reduce_mean(self._labeled_loss(self.x_labeled, self.labels))
        L_u = tf.reduce_mean(self._unlabeled_loss(self.x_unlabeled))
        L_e = tf.reduce_mean(self._qxy_loss(self.x_labeled, self.labels))
	weight_prior = self._weight_regularization()
        self.loss = -(L_l + L_u + self.alpha*L_e - 0.5 * weight_prior)
        
	## define optimizer
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss, global_step=self.global_step)
	
        ## compute accuracies
	train_acc = self.compute_acc(self.x_train, self.y_train)
	test_acc = self.compute_acc(self.x_test, self.y_test)
	self._create_summaries(L_l, L_u, L_e)

        ## initialize session and train
        max_acc, epoch, step = 0, 0, 0
	with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            total_loss, l_l, l_u, l_e = 0.0, 0.0, 0.0, 0.0
	    saver = tf.train.Saver()
	    if self.LOGGING:
                writer = tf.summary.FileWriter(self.LOGDIR, sess.graph)

            while epoch < self.NUM_EPOCHS:
                x_labeled, labels, x_unlabeled, _ = Data.next_batch(self.LABELED_BATCH_SIZE, self.UNLABELED_BATCH_SIZE)
		if self.BINARIZE == True:
	 	    x_labeled, x_unlabeled = self._binarize(x_labeled), self._binarize(x_unlabeled)

	 	fd = {self.x_labeled:x_labeled, self.x_unlabeled:x_unlabeled, self.labels:labels}
            	_, loss_batch, l_lb, l_ub, l_eb, summary_elbo = sess.run([self.optimizer, self.loss, L_l, L_u, L_e, self.summary_op_elbo], 
            			     	     		    feed_dict={self.x_labeled: x_labeled, 
		           		    	 		       self.labels: labels, self.beta: self.schedule[epoch],
		  	            		     		       self.x_unlabeled: x_unlabeled,
								       self.beta:self.schedule[epoch]})
            	if self.LOGGING:
             	    writer.add_summary(summary_elbo, global_step=step)
		total_loss, l_l, l_u, l_e, step = total_loss+loss_batch, l_l+l_lb, l_u+l_ub, l_e+l_eb, step+1

                if Data._epochs_unlabeled > epoch:
		    fd = self._printing_feed_dict(Data, x_labeled, xlabels)
		    acc_train, acc_test, summary_acc = sess.run([self.train_acc, self.test_acc, self.summary_op_acc],fd)

        	    self._save_model(saver, sess, step, max_acc, acc_test)
		    max_acc = acc_test if acc_test > max_acc else max_acc
		    if self.LOGGING:
			writer.add_summary(summary_acc, global_step=epoch)

		    if self.verbose==0:
                        """ Print ELBOs and accuracy"""
                        self._print_verbose0(epoch, step, total_loss, l_l, l_u, l_e, acc_train, acc_test)
                        total_loss, l_l, l_u, l_e, step = 0.0, 0.0, 0.0, 0.0, 0

                    elif self.verbose==1:
                        """ Print semi-supervised aspects """
                        self._print_verbose1(epoch, fd, sess)

                    elif self.verbose==2:
                        acc_train, acc_test,  = sess.run([train_acc_q, test_acc_q], feed_dict=fd)
                        print('At epoch {}: Training: {:5.3f}, Test: {:5.3f}'.format(epoch, acc_train, acc_test))


                    epoch += 1
            if self.LOGGING:
                writer.close()

    

    def predict_new(self, x, n_iters=100):
	predictions = self.predict(x, n_iters)
	saver = tf.train.Saver()
	with tf.Session() as session:
            ckpt = tf.train.get_checkpoint_state(self.ckpt_dir)
            saver.restore(session, ckpt.model_checkpoint_path)
            preds = session.run([predictions])
	return preds[0]



    def predict(self, x, n_iters=100):
	z_, _ = dgm._forward_pass_Gauss(x, self.Qx_z, self.NUM_HIDDEN, self.NONLINEARITY)
	h = tf.concat([x, z_], axis=1)
	return dgm._forward_pass_Cat(h, self.Pzx_y, self.NUM_HIDDEN, self.NONLINEARITY)



    def _sample_xy(self, n_samples=int(1e3)):
	saver = tf.train.Saver()
	with tf.Session() as session:
            ckpt = tf.train.get_checkpoint_state(self.ckpt_dir)
            saver.restore(session, ckpt.model_checkpoint_path)
            z_ = np.random.normal(size=(n_samples, self.Z_DIM)).astype('float32')
            if self.TYPE_PX=='Gaussian':
                x_ = dgm._forward_pass_Gauss(z_, self.Pz_x, self.NUM_HIDDEN, self.NONLINEARITY)
            else:
            	x_ = dgm._forward_pass_Bernoulli(z_, self.Pz_x, self.NUM_HIDDEN, self.NONLINEARITY)
            h = tf.concat([x_[0],z_], axis=1)
            y_ = dgm._forward_pass_Cat(h, self.Pzx_y, self.NUM_HIDDEN, self.NONLINEARITY)
            x,y = session.run([x_,y_])
        return x[0],y

    def _sample_Z(self, x, n_samples):
	""" Sample from Z with the reparamterization trick """
	mean, log_var = dgm._forward_pass_Gauss(x, self.Qx_z, self.NUM_HIDDEN, self.NONLINEARITY)
	eps = tf.random_normal([tf.shape(x)[0], self.Z_DIM], 0, 1, dtype=tf.float32)
	return mean, log_var, mean + tf.sqrt(tf.exp(log_var)) * eps


    def _labeled_loss(self, x, y, z=None, q_mean=None, q_logvar=None):
	""" Compute necessary terms for labeled loss (per data point) """
	if z==None:
	    q_mean, q_logvar, z = self._sample_Z(x, self.Z_SAMPLES)
        logpx = self._compute_logpx(x, z)
	logpy = self._compute_logpy(y, x, z)
	klz = dgm._gauss_kl(q_mean, q_logvar)
	return logpx + logpy - self.beta * klz


    def _unlabeled_loss(self, x):
	""" Compute necessary terms for unlabeled loss (per data point) """
	weights = dgm._forward_pass_Cat(x, self.Qx_y, self.NUM_HIDDEN, self.NONLINEARITY)
	q_mean, q_log_var, z = self._sample_Z(x, self.Z_SAMPLES)
	EL_l = 0 
	for i in range(self.NUM_CLASSES):
	    y = self._generate_class(i, x.get_shape()[0])
	    EL_l += tf.multiply(weights[:,i], self._labeled_loss(x, y, z, q_mean, q_log_var))
	ent_qy = -tf.reduce_sum(tf.multiply(weights, tf.log(1e-10 + weights)), axis=1)
	return EL_l + ent_qy


    def _qxy_loss(self, x, y):
	y_ = dgm._forward_pass_Cat_logits(x, self.Qx_y, self.NUM_HIDDEN, self.NONLINEARITY)
	return -tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_)


    def _vae_loss(self, x):
 	z_mean, z_log_var, z = self._sample_Z(x,1)
        KLz = dgm._gauss_kl(z_mean, tf.exp(z_log_var))
        l_qz = dgm._gauss_logp(z, z_mean, tf.exp(z_log_var))
        l_pz = dgm._gauss_logp(z, tf.zeros_like(z), tf.ones_like(z))
        l_px = self._compute_logpx(x, z)
        total_elbo = l_px + self.beta * (l_pz - l_qz)
        return tf.reduce_sum(total_elbo), tf.reduce_mean(l_px), tf.reduce_mean(KLz)


    def compute_acc(self, x, y):
	y_ = self.predict(x)
	return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_,axis=1), tf.argmax(y, axis=1)), tf.float32))

    def _initialize_networks(self):
    	""" Initialize all model networks """
	if self.TYPE_PX == 'Gaussian':
      	    self.Pz_x = dgm._init_Gauss_net(self.Z_DIM, self.NUM_HIDDEN, self.X_DIM, 'Pz_x_')
	elif self.TYPE_PX == 'Bernoulli':
	    self.Pz_x = dgm._init_Cat_net(self.Z_DIM, self.NUM_HIDDEN, self.X_DIM, 'Pz_x_')
    	self.Pzx_y = dgm._init_Cat_net(self.Z_DIM+self.X_DIM, self.NUM_HIDDEN, self.NUM_CLASSES, 'Pzx_y_')
    	self.Qx_z = dgm._init_Gauss_net(self.X_DIM, self.NUM_HIDDEN, self.Z_DIM, 'Qx_z_')
    	self.Qx_y = dgm._init_Cat_net(self.X_DIM, self.NUM_HIDDEN, self.NUM_CLASSES, 'Qx_y_')

    

    def _generate_class(self, k, num):
	""" create one-hot encoding of class k with length num """
	y = np.zeros(shape=(num, self.NUM_CLASSES))
	y[:,k] = 1
	return tf.constant(y, dtype=tf.float32)

