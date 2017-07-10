from __future__ import absolute_import
from __future__ import division 
from __future__ import print_function

import sys, os, pdb
parent_dir = os.getcwd()
path = os.path.dirname(parent_dir)
sys.path.append(path)

from models.model import model

import numpy as np

import utils.dgm as dgm
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

import pdb

""" Implementation of Kingma et al (2014), M2:  P(Z)P(Y)P(X|Y,Z) """

class M2(model):
   
    def __init__(self, Z_DIM=2, LEARNING_RATE=0.005, NUM_HIDDEN=4, ALPHA=0.1, NONLINEARITY=tf.nn.relu, TYPE_PX='Gaussian', start_temp=0.8, BINARIZE=False,
 		 temperature_epochs=None, LABELED_BATCH_SIZE=16, UNLABELED_BATCH_SIZE=128, NUM_EPOCHS=75, Z_SAMPLES=1, verbose=1, logging=True):
    	
	super(M2, self).__init__(Z_DIM, LEARNING_RATE, NUM_HIDDEN, TYPE_PX, NONLINEARITY, temperature_epochs, start_temp, NUM_EPOCHS, Z_SAMPLES, BINARIZE, logging)

    	self.LABELED_BATCH_SIZE = LABELED_BATCH_SIZE         # labeled batch size 
	self.UNLABELED_BATCH_SIZE = UNLABELED_BATCH_SIZE     # labeled batch size 
    	self.alpha = ALPHA 				     # weighting for additional term
    	self.verbose = verbose				     # control output: 1 for ELBO, else accuracy
	self.name = 'm2'                                     # model name
    

    def fit(self, Data):
    	self._process_data(Data)
    	
	self._create_placeholders() 
	self._set_schedule()
        self._initialize_networks()
        
        ## define loss function
	self._compute_loss_weights()
        L_l = tf.reduce_mean(self._labeled_loss(self.x_labeled, self.labels))
        L_u = tf.reduce_mean(self._unlabeled_loss(self.x_unlabeled))
        L_e = tf.reduce_mean(self._qxy_loss(self.x_labeled, self.labels))
	weight_priors = self._weight_regularization() / (self.LABELED_BATCH_SIZE + self.UNLABELED_BATCH_SIZE)
        self.loss = -(L_l + L_u + self.alpha * L_e - 0.8 * weight_priors)
	
        ## define optimizer
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss, global_step=self.global_step)
	
	## compute accuracies
	self.train_acc = self.compute_acc(self.x_train, self.y_train)
	self.test_acc = self.compute_acc(self.x_test, self.y_test)
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
                x_labeled, labels, x_unlabeled, _ = Data.next_batch(self.LABELED_BATCH_SIZE, self.UNLABELED_BATCH_SIZE)
                if self.BINARIZE == True:
                    x_labeled, x_unlabeled = self._binarize(x_labeled), self._binarize(x_unlabeled)
                _, loss_batch, l_lb, l_ub, l_eb, summary_elbo = sess.run([self.optimizer, self.loss, L_l, L_u, L_e, self.summary_op_elbo],
                                                            feed_dict={self.x_labeled: x_labeled,
                                                                       self.labels: labels, self.beta:self.schedule[epoch],
                                                                       self.x_unlabeled: x_unlabeled})          	
		if self.LOGGING:
		    writer.add_summary(summary_elbo, global_step=self.global_step)
                total_loss, l_l, l_u, l_e, step = total_loss+loss_batch, l_l+l_lb, l_u+l_ub, l_e+l_eb, step+1

                if Data._epochs_labeled > epoch:
		    fd = self._printing_feed_dict(Data, x_labeled, labels)
		    acc_test, summary_acc = sess.run([self.test_acc, self.summary_op_acc], fd)
	
 		    self._save_model(saver, sess, step, max_acc, acc_test)
		    max_acc = acc_test if acc_test > max_acc else max_acc

		    if self.verbose == 0:
		    	self._print_verbose0(epoch, step, total_loss, l_l, l_u, l_e)
        	        total_loss, l_l, l_u, l_e, step = 0.0, 0.0, 0.0, 0.0, 0
        	    
		    elif self.verbose == 1:
            	        self._print_verbose1(epoch, fd, sess)
	    	    epoch+=1
	    if self.LOGGING:
	        writer.close()

    def predict_new(self, x, n_iters=100):
        predictions = self.predict(x)
        saver = tf.train.Saver()
        with tf.Session() as session:
            ckpt = tf.train.get_checkpoint_state(self.ckpt_dir)
            saver.restore(session, ckpt.model_checkpoint_path)
            preds = session.run([predictions])
        return preds[0]


    def predict(self, x):
	return dgm._forward_pass_Cat(x, self.Qx_y, self.NUM_HIDDEN, self.NONLINEARITY)



    def _sample_xy(self, n_samples=int(1e3)):
        saver = tf.train.Saver()
        with tf.Session() as session:
            ckpt = tf.train.get_checkpoint_state(self.ckpt_dir)
            saver.restore(session, ckpt.model_checkpoint_path)

            z_ = np.random.normal(size=(n_samples, self.Z_DIM)).astype('float32')
	    p = np.ones(self.NUM_CLASSES)*(1/self.NUM_CLASSES)
	    y_ = np.random.multinomial(1,p, size=n_samples).astype('float32')
	    h = tf.concat([z_,y_], axis=1)
            if self.TYPE_PX=='Gaussian':
                mean, logvar = dgm._forward_pass_Gauss(h, self.Pzy_x, self.NUM_HIDDEN, self.NONLINEARITY)
		eps = tf.random_normal([n_samples, self.X_DIM], dtype=tf.float32)
		x_ = mean + tf.sqrt(tf.exp(logvar)) * eps
            else:
                x_ = dgm._forward_pass_Bernoulli(z_, self.Pzy_x, self.NUM_HIDDEN, self.NONLINEARITY)
            x = session.run([x_])
        return x[0],y_



    def _sample_Z(self, x, y, n_samples):
	""" Sample from Z with the reparamterization trick """
	h = tf.concat([x, y], axis=1)
	mean, log_var = dgm._forward_pass_Gauss(h, self.Qxy_z, self.NUM_HIDDEN, self.NONLINEARITY)
	eps = tf.random_normal([tf.shape(x)[0], self.Z_DIM], 0, 1, dtype=tf.float32)
	return mean, log_var, mean + tf.sqrt(tf.exp(log_var)) * eps


    def _labeled_loss(self, x, y):
	""" Compute necessary terms for labeled loss (per data point) """
	z_mean, z_log_var, z  = self._sample_Z(x, y, self.Z_SAMPLES)
	l_px = self._compute_logpx(x, z, y)
	l_py = self._compute_logpy(y)       
	l_pz = dgm._gauss_logp(z, tf.zeros_like(z), tf.log(tf.ones_like(z)))
	l_qz = dgm._gauss_logp(z, z_mean, z_log_var)
	KLz = dgm._gauss_kl(z_mean, z_log_var)
	return l_px + l_py + self.beta * (l_pz - l_qz)

    def _unlabeled_loss(self, x):
	""" Compute necessary terms for unlabeled loss (per data point) """
	weights = dgm._forward_pass_Cat(x, self.Qx_y, self.NUM_HIDDEN, self.NONLINEARITY)
	EL_l = 0 
	for i in range(self.NUM_CLASSES):
	    y = self._generate_class(i, x.get_shape()[0])
	    EL_l += tf.multiply(weights[:,i], self._labeled_loss(x, y))
	ent_qy = -tf.reduce_sum(tf.multiply(weights, tf.log(weights+1e-10)), axis=1)
	return tf.add(EL_l, ent_qy)


    def _qxy_loss(self, x, y):
	y_ = dgm._forward_pass_Cat_logits(x, self.Qx_y, self.NUM_HIDDEN, self.NONLINEARITY)
	return -tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_)


    def _compute_logpx(self, x, z, y):
	""" compute the likelihood of every element in x under p(x|z,y) """
	h = tf.concat([z,y], axis=1)
	if self.TYPE_PX == 'Gaussian':
	    mean, log_var = dgm._forward_pass_Gauss(h, self.Pzy_x, self.NUM_HIDDEN, self.NONLINEARITY)
	    return dgm._gauss_logp(x, mean, log_var)
	elif self.TYPE_PX == 'Bernoulli':
	    logits = dgm._forward_pass_Cat_logits(h, self.Pzy_x, self.NUM_HIDDEN, self.NONLINEARITY)
	    return -tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=x, logits=logits), axis=1)


    def _compute_logpy(self, y, x=None, z=None):
	""" compute the likelihood of every element in y under p(y) """
	return -tf.reduce_mean(tf.multiply(y, tf.log(self.Py + 1e-10)), axis=1)


    def compute_acc(self, x, y):
	y_ = self.predict(x)
	return  tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_,axis=1), tf.argmax(y, axis=1)), tf.float32))


    def _initialize_networks(self):
    	""" Initialize all model networks """
	if self.TYPE_PX == 'Gaussian':
    	    self.Pzy_x = dgm._init_Gauss_net(self.Z_DIM + self.NUM_CLASSES, self.NUM_HIDDEN, self.X_DIM, 'Pzy_x')
    	elif self.TYPE_PX =='Bernoulli':
    	    self.Pzy_x = dgm._init_Cat_net(self.Z_DIM + self.NUM_CLASSES, self.NUM_HIDDEN, self.X_DIM, 'Pzy_x')	
    	self.Py = tf.constant((1./self.NUM_CLASSES)*np.ones(shape=(self.NUM_CLASSES,)), dtype=tf.float32)
    	self.Qxy_z = dgm._init_Gauss_net(self.X_DIM+self.NUM_CLASSES, self.NUM_HIDDEN, self.Z_DIM, 'Qxy_z')
    	self.Qx_y = dgm._init_Cat_net(self.X_DIM, self.NUM_HIDDEN, self.NUM_CLASSES, 'Qx_y')

    
    def _generate_class(self, k, num):
	""" create one-hot encoding of class k with length num """
	y = np.zeros(shape=(num, self.NUM_CLASSES))
	y[:,k] = 1
	return tf.constant(y, dtype=tf.float32)


    def _print_verbose1(self, epoch, fd, sess):
	zm_test, zlv_test, z_test = self._sample_Z(self.x_test,self.y_test,1)
        zm_train, zlv_train, z_train = self._sample_Z(self.x_train,self.y_train,1)
        lpx_test, lpx_train,klz_test,klz_train, acc_train, acc_test = sess.run([self._compute_logpx(self.x_test, z_test, self.y_test),
                                                                  self._compute_logpx(self.x_train, z_train, self.y_train),
                                                                  dgm._gauss_kl(zm_test, tf.exp(zlv_test)),
                                                                  dgm._gauss_kl(zm_train, tf.exp(zlv_train)),
                                                                  self.train_acc, self.test_acc], feed_dict=fd)
	print('Epoch: {}, logpx: {:5.3f}, klz: {:5.3f}, Train: {:5.3f}, Test: {:5.3f}'.format(epoch, np.mean(lpx_train), np.mean(klz_train), acc_train, acc_test ))



