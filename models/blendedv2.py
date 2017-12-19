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
Implementation of blended semi-supervised model: [ p(z) * p(y) * p(x|y,z) ] * p(y|x)
Inference network: q(z|x, y) 
Prior over theta, theta_1 ties recognition network (generative model) and discriminator together
p(theta, theta_1) = p(theta) * p(theta_1) * N(theta_1; theta, sigma(alpha)) 
"""

class blended(model):
   
    def __init__(self, n_x, n_y, n_z=2, n_hid=[4], alpha=0.1, beta=None, x_dist='Gaussian', nonlinearity=tf.nn.relu, batchnorm=False, l2_reg=0.3, mc_samples=1,ckpt=None):
	
	if beta is None:
	    beta = tf.get_variable(name='beta', shape=[], dtype=tf.float32, initializer=tf.constant([0.1]))
	self.sigma = ( beta / (1-beta))**2.0
	self.reg_term = tf.placeholder(tf.float32, shape=[], name='reg_term')
	super(blended, self).__init__(n_x, n_y, n_z, n_hid, x_dist, nonlinearity, batchnorm, mc_samples, alpha, l2_reg, ckpt)

	""" TODO: add any general terms we want to have here """
	self.name = 'blended'

    def build_model(self):
	""" Define model components and variables """
	self.create_placeholders()
	self.initialize_networks()
	## model variables and relations ##
	# inference #
        self.y_ = dgm.forwardPassCatLogits(self.x, self.qy_x, self.n_hid, self.nonlinearity, self.bn, scope='qy_x', reuse=False)
	self.qz_in = tf.concat([self.x, self.y], axis=-1) 	
	self.qz_mean, self.qz_lv, self.z_ = dgm.samplePassGauss(self.qz_in, self.qz_xy, self.n_hid, self.nonlinearity, self.bn, scope='qz_xy', reuse=False)
	# generative #
	self.z_prior = tf.random_normal([tf.shape(self.y)[0], self.n_z])
	self.px_in = tf.concat([self.y, self.z_prior], axis=-1)
	if self.x_dist == 'Gaussian':
	    self.px_mean, self.px_lv, self.x_ = dgm.samplePassGauss(self.px_in, self.px_yz, self.n_hid, self.nonlinearity, self.bn, scope='px_yz', reuse=False)
	    self.x_ = tf.reshape(self.x_, [-1, self.n_x])
	elif self.x_dist == 'Bernoulli':
	    self.x_ = dgm.forwardPassBernoulli(self.px_in, self.px_yz, self.n_hid, self.nonlinearity, self.bn, scope='px_yz', reuse=False)
	self.py = dgm.forwardPassCat(self.x_, self.py_x, self.n_hid, self.nonlinearity, self.bn, scope='py_x', reuse=False)
	self.predictions = self.predict(self.x, training=False)

    def compute_loss(self):
	""" manipulate computed components and compute loss """
	self.elbo_l = tf.reduce_mean(self.labeled_loss(self.x_l, self.y_l))
	self.elbo_u = tf.reduce_mean(self.unlabeled_loss(self.x_u))
	self.weight_priors = self.compute_prior() 
	return -(self.elbo_l + self.elbo_u + self.weight_priors)

    def labeled_loss(self, x, y):
	""" condition on label (recover M2) """
	z_m, z_lv, z = self.sample_z(x,y)
	x_ = tf.tile(tf.expand_dims(x,0),[self.mc_samples,1,1])
	y_ = tf.tile(tf.expand_dims(y,0),[self.mc_samples,1,1])
	g_term = self.lowerBound(x_,y_,z,z_m,z_lv)
	d_term = self.discriminatorTerm(x,y)
	return g_term + self.alpha * d_term
 
    def discriminatorTerm(self, x, y):
	y_ = dgm.forwardPassCatLogits(x, self.py_x, self.n_hid, self.nonlinearity, self.bn, scope='py_x')
	return dgm.multinoulliLogDensity(y, y_)

    def unlabeled_loss(self, x):
	n_u = tf.shape(x)[0] 
	qy_l = dgm.forwardPassCat(x, self.qy_x, self.n_hid, self.nonlinearity, self.bn, scope='qy_x')
	x_r = tf.tile(x, [self.n_y,1])
	y_u = tf.reshape(tf.tile(tf.eye(self.n_y), [1, n_u]), [-1, self.n_y])
	z_m, z_lv, z = self.sample_z(x_r,y_u)
	x_ = tf.tile(tf.expand_dims(x_r,0),[self.mc_samples,1,1]) 
	y_ = tf.tile(tf.expand_dims(y_u,0),[self.mc_samples,1,1])
	lb_u = tf.transpose(tf.reshape(self.lowerBound(x_, y_, z, z_m, z_lv), [self.n_y, n_u]))
	lb_u = tf.reduce_sum(qy_l * lb_u, axis=-1)
	qy_entropy = -tf.reduce_sum(qy_l * tf.log(qy_l + 1e-10), axis=-1)
	return lb_u + qy_entropy

    def lowerBound(self, x, y, z, z_m, z_lv):
	""" Compute densities and lower bound given all inputs (mc_samps X n_obs X n_dim) """
	l_px = self.compute_logpx(x,y,z)
	l_py = dgm.multinoulliUniformLogDensity(y)
	l_pz = dgm.standardNormalLogDensity(z)
	l_qz = dgm.gaussianLogDensity(z, z_m, z_lv)
	return tf.reduce_mean(l_px + l_py + l_pz - l_qz, axis=0)
	
    def sample_z(self, x, y):
	l_qz_in = tf.concat([x, y], axis=-1)
        z_m, z_lv, z =  dgm.samplePassGauss(l_qz_in, self.qz_xy, self.n_hid, self.nonlinearity, self.bn, mc_samps=self.mc_samples, scope='qz_xy')
	return tf.tile(tf.expand_dims(z_m,0), [self.mc_samples,1,1]), tf.tile(tf.expand_dims(z_lv,0),[self.mc_samples,1,1]), z

    def compute_logpx(self, x, y, z):
	px_in = tf.reshape(tf.concat([y,z], axis=-1), [-1, self.n_y+self.n_z])
	if self.x_dist == 'Gaussian':
            mean, log_var = dgm.forwardPassGauss(px_in, self.px_yz, self.n_hid, self.nonlinearity, self.bn, scope='px_yz')
	    mean, log_var = tf.reshape(mean, [self.mc_samples, -1, self.n_x]),  tf.reshape(log_var, [self.mc_samples, -1, self.n_x])
            return dgm.gaussianLogDensity(x, mean, log_var)
        elif self.x_dist == 'Bernoulli':
            logits = dgm.forwardPassCatLogits(px_in, self.px_yz, self.n_hid, self.nonlinearity, self.bn, scope='px_yz')
	    logits = tf.reshape(logits, [self.mc_samples, -1, self.n_x])
            return dgm.bernoulliLogDensity(x, logits) 

    def compute_prior(self):
	""" compute the log prior term """
	weights = [V for V in tf.trainable_variables() if 'py_x' not in V.name]	
	diffs = [V for V in tf.trainable_variables() if 'eps' in V.name]	
	weight_term = np.sum([tf.reduce_sum(dgm.standardNormalLogDensity(w)) for w in weights])
	diff_term = -tf.log(self.sigma) - np.sum([tf.nn.l2_loss(w) for w in diffs])/(self.sigma**2)
 	return self.l2_reg * ( weight_term + diff_term ) / self.reg_term
   
    def predict(self, x, training=True):
	""" predict y for given x with p(y|x) """
	return dgm.forwardPassCat(x, self.py_x, self.n_hid, self.nonlinearity, self.bn, training=training, scope='py_x') 

    def predictq(self, x):
	""" predict y for given x with q(y|x) """
	return dgm.forwardPassCat(x, self.qy_x, self.n_hid, self.nonlinearity, self.bn, scope='qy_x') 

    def encode(self, x, y=None, n_iters=100):
	""" encode a new example into z-space (labeled or unlabeled) """
	if y is None:
	    y = tf.one_hot(tf.argmax(dgm.forwardPassCat(x, self.qy_x, self.n_hid, self.nonlinearity, self.bn, scope='qy_x'), axis=1), self.n_y)
	_, _, z = self.sample_z(x, y)
	return z

    def compute_acc(self, x, y):
	y_ = self.predict(x)
	acc =  tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_,axis=1), tf.argmax(y, axis=1)), tf.float32))
	return acc 

    def compute_accq(self, x, y):
	y_ = self.predictq(x)
	acc =  tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_,axis=1), tf.argmax(y, axis=1)), tf.float32))
	return acc 

    def initialize_networks(self):
    	""" Initialize all model networks """
	if self.x_dist == 'Gaussian':
      	    self.px_yz = dgm.initGaussNet(self.n_z+self.n_y, self.n_hid, self.n_x, 'px_yz_')
	elif self.x_dist == 'Bernoulli':
	    self.px_yz = dgm.initCatNet(self.n_z+self.n_y, self.n_hid, self.n_x, 'px_yz_')
    	self.qz_xy = dgm.initGaussNet(self.n_x+self.n_y, self.n_hid, self.n_z, 'qz_xy_')
    	self.qy_x = dgm.initCatNet(self.n_x, self.n_hid, self.n_y, 'qy_x_') # recognition network
	self.py_x = dgm.initTiedNetwork(self.qy_x, self.n_hid, 'py_x_', 'Categorical') # discriminator tied to recognition network

    def training_fd(self, x_l, y_l, x_u):
	return {self.x_l: x_l, self.y_l: y_l, self.x_u: x_u, self.x: x_l, self.y: y_l, self.reg_term:self.n_train}

    def _printing_feed_dict(self, Data, x_l, x_u, y, eval_samps, binarize):
	fd = super(blended,self)._printing_feed_dict(Data, x_l, x_u, y, eval_samps, binarize)
	fd[self.reg_term] = self.n_train
	return fd

    def print_verbose1(self, epoch, fd, sess):
	total, elbo_l, elbo_u = sess.run([self.compute_loss(), self.elbo_l, self.elbo_u] ,fd)
	train_acc, test_acc = sess.run([self.train_acc, self.test_acc], fd)
	print("Epoch {}: Total: {:5.3f}, Labeled: {:5.3f}, Unlabeled: {:5.3f}, Training: {:5.3f}, Testing: {:5.3f}".format(epoch, total, elbo_l, elbo_u, train_acc, test_acc))	

    def print_verbose2(self, epoch, fd, sess):
	total, elbo_l, elbo_u = sess.run([self.compute_loss(), self.elbo_l, self.elbo_u] ,fd)
	train_acc, test_acc = sess.run([self.train_acc, self.test_acc], fd)
	trainq = sess.run(self.compute_accq(self.x_train, self.y_train), fd)
	testq = sess.run(self.compute_accq(self.x_test, self.y_test), fd)
	print("Epoch {}: Total: {:5.3f}, Labeled: {:5.3f}, Unlabeled: {:5.3f}, Training (p): {:5.3f}, Testing (p): {:5.3f} "
 		"Training (q): {:5.3f}, Testing (q): {:5.3f}".format(epoch, total, elbo_l, elbo_u, train_acc, test_acc, trainq, testq))	
