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
Implementation of auxiliary DGMs from Maaloe et al. (2016): p(z) * p(y) * p(x|y,z) * p(a|z,y,x)
Inference network: q(a,z,y|x) = q(a|x) * q(y|a,x) * q(z|a,y,x) 
"""

class adgm(model):
   
    def __init__(self, n_x, n_y, n_z=2, n_a=2, n_hid=[4], alpha=0.1, x_dist='Gaussian', nonlinearity=tf.nn.relu, batchnorm=False, mc_samples=1,ckpt=None):
	
	self.n_a = n_a        # auxiliary variable dimension
	self.name = 'adgm'
    	
	super(adgm, self).__init__(n_x, n_y, n_z, n_hid, x_dist, nonlinearity, batchnorm, mc_samples, alpha, ckpt)


	""" TODO: add any general terms we want to have here """
	self.train_acc = self.compute_acc(self.x_train, self.y_train)
	self.test_acc = self.compute_acc(self.x_test, self.y_test)
	self.predictions = self.predict(self.x_new)

    def build_model(self):
	""" Define model components and variables """
	self.create_placeholders()
	self.initialize_networks()

	### labeled components ###
	## recognition q(a|x) ##
	self.qa_mean_l, self.qa_lv_l, self.a_l = dgm.samplePassGauss(self.x_l, self.qa_x, self.n_hid, self.nonlinearity, self.bn, True, 'qa_x')
	## recognition q(z|x,y,a)## 
	self.l_qz_in = tf.concat([self.x_l, self.y, self.a_l], axis=1)
	self.qz_mean_l, self.qz_lv_l, self.z_l = dgm.samplePassGauss(self.l_qz_in, self.qz_xya, self.n_hid, self.nonlinearity, self.bn, True, 'qz_xya')
	## classifier q(y|x,a) ##
	self.l_qy_in = tf.concat([self.x_l, self.a_l], axis=1)
	self.l_qy_l = dgm.forwardPassCatLogits(self.l_qy_in, self.qy_xa, self.n_hid, self.nonlinearity, self.bn, True, 'qy_xa')
	## generative p(x|z,y) ##
	self.l_px_in = tf.concat([self.y, self.z_l], axis=1)
	if self.x_dist == 'Gaussian':
	    self.px_mean_l, self.px_lv_l = dgm.forwardPassGauss(self.l_px_in, self.px_yz, self.n_hid, self.nonlinearity, self.bn, True, 'px_yz')	    
	elif self.x_dist == 'Bernoulli':
	    self.l_px_l = dgm.forwardPassCatLogits(self.l_px_in, self.px_yz, self.n_hid, self.nonlinearity, self.bn, True, 'px_yz')
	## generative p(a|x,y,z) ##
	self.l_pa_in = tf.concat([self.x_l, self.y, self.z_l], axis=1)
	self.pa_mean_l, self.pa_lv_l = dgm.forwardPassGauss(self.l_pa_in, self.pa_xyz, self.n_hid, self.nonlinearity, self.bn, True, 'pa_xyz')

	### unlabeled components ###
	# first compute a_u since it is out of y expectation 
	self.qa_mean_u, self.qa_lv_u, self.a_u = dgm.samplePassGauss(self.x_u, self.qa_x, self.n_hid, self.nonlinearity, self.bn, True, 'qa_x')
	# tile x_u, a_u parameters, and generate y_u
	self.xu_tiled, self.au_tiled = tf.tile(self.x_u, [self.n_y, 1]), tf.tile(self.a_u, [self.n_y, 1])
	self.qa_mean_tiled, self.qa_lv_tiled = tf.tile(self.qa_mean_u, [self.n_y, 1]), tf.tile(self.qa_lv_u, [self.n_y, 1])	
	self.y_u = tf.reshape(tf.tile(tf.eye(self.n_y), [1, tf.shape(self.x_u)[0]]), [-1, self.n_y])
	## recognition q(z|x,y,a)## 
	self.u_qz_in = tf.concat([self.xu_tiled, self.y_u, self.au_tiled], axis=1)
	self.qz_mean_u, self.qz_lv_u, self.z_u = dgm.samplePassGauss(self.u_qz_in, self.qz_xya, self.n_hid, self.nonlinearity, self.bn, True, 'qz_xya')
	## classifier q(y|x,a) ##
	self.u_qy_in = tf.concat([self.x_u, self.a_u], axis=1)
	self.qy = dgm.forwardPassCat(self.u_qy_in, self.qy_xa, self.n_hid, self.nonlinearity, self.bn, True, 'qy_xa')
	## generative p(x|z,y) ##
	self.u_px_in = tf.concat([self.y_u, self.z_u], axis=1)
	if self.x_dist == 'Gaussian':
	    self.px_mean_u, self.px_lv_u = dgm.forwardPassGauss(self.u_px_in, self.px_yz, self.n_hid, self.nonlinearity, self.bn, True, 'px_yz')	    
	elif self.x_dist == 'Bernoulli':
	    self.l_px_u = dgm.forwardPassCatLogits(self.u_px_in, self.px_yz, self.n_hid, self.nonlinearity, self.bn, True, 'px_yz')
	## generative p(a|x,y,z) ##
	self.u_pa_in = tf.concat([self.xu_tiled, self.y_u, self.z_u], axis=1)
	self.pa_mean_u, self.pa_lv_u = dgm.forwardPassGauss(self.u_pa_in, self.pa_xyz, self.n_hid, self.nonlinearity, self.bn, True, 'pa_xyz')

    def compute_loss(self):
	""" manipulate computed components and compute loss """
	l_lb = self.labeled_loss()
	l_ub = self.unlabeled_loss()
	qy_loglik = dgm.multinoulliLogDensity(self.y, self.l_qy_l)
	weight_priors = self.weight_regularization()
	self.elbo_l = tf.reduce_mean(l_lb)
	self.elbo_u = tf.reduce_mean(l_ub)
	return (tf.reduce_mean(l_lb + l_ub + self.alpha * qy_loglik)*self.n - weight_priors)/(-self.n)

    def labeled_loss(self):
	""" Compute necessary terms for labeled loss (per data point) """
	l_pz = dgm.standardNormalLogDensity(self.z_l)
	l_py = dgm.multinoulliUniformLogDensity(self.y)
	if self.x_dist == 'Gaussian':
	    l_px = dgm.gaussianLogDensity(self.x_l, self.px_mean_l, self.px_lv_l)
	elif self.x_dist == 'Bernoulli':
	    l_px = dgm.bernoulliLogDensity(self.x_l, self.l_px_l)
	l_pa = dgm.gaussianLogDensity(self.a_l, self.pa_mean_l, self.pa_lv_l)
	l_qz = dgm.gaussianLogDensity(self.z_l, self.qz_mean_l, self.qz_lv_l)
	l_qa = dgm.gaussianLogDensity(self.a_l, self.qa_mean_l, self.qa_lv_l)
	return l_px + l_py + l_pz + l_pa - l_qz - l_qa 

    def unlabeled_loss(self):
	""" Compute necessary terms for unlabeled loss (per data point) """
	### compute relevant densities ##
	l_pz = dgm.standardNormalLogDensity(self.z_u)
	l_py = dgm.multinoulliUniformLogDensity(self.y_u)
	if self.x_dist == 'Gaussian':
	    l_px = dgm.gaussianLogDensity(self.xu_tiled, self.px_mean_u, self.px_lv_u)
	elif self.x_dist == 'Bernoulli':
	    l_px = dgm.bernoulliLogDensity(self.xu_tiled, self.l_px_u)
	l_pa = dgm.gaussianLogDensity(self.au_tiled, self.pa_mean_u, self.pa_lv_u)
	l_qz = dgm.gaussianLogDensity(self.z_u, self.qz_mean_u, self.qz_lv_u)
	l_qa = dgm.gaussianLogDensity(self.au_tiled, self.qa_mean_tiled, self.qa_lv_tiled)
	### sum densities and reshape them into n_u x n_y matrix ###
	n_u = tf.shape(self.x_u)[0] 
	densities = tf.transpose(tf.reshape(l_px + l_py + l_pz + l_pa - l_qz - l_qa,  [self.n_y, n_u]))
	### take expectations w.r.t. q(y|x,a) and add entropy ###
	qy_entropy = -tf.reduce_sum(self.qy * tf.log(1e-10 + self.qy), axis=1)
	return tf.reduce_sum(self.qy * densities, axis=1) + qy_entropy

    def predict(self, x):
	""" predict y for given x with q(y|x,a) """
	_, _, a = dgm.samplePassGauss(x, self.qa_x, self.n_hid, self.nonlinearity, self.bn, True, 'qa_x')
	h = tf.concat([x,a], axis=1)
	return dgm.forwardPassCat(h, self.qy_xa, self.n_hid, self.nonlinearity, self.bn, True, 'qy_xa'), a

    def encode(self, x, y=None, n_iters=100):
	""" encode a new example into z-space (labeled or unlabeled) """
	_, _, a = dgm.samplePassGauss(x, self.qa_x, self.n_hid, self.nonlinearity, self.bn, True, 'qa_x')
	if y is None:
	    h = tf.concat([x,a], axis=1)
	    y = tf.one_hot(tf.argmax(dgm.forwardPassCat(h, self.qy_xa, self.n_hid, self.nonlinearity, self.bn, True, 'qa_x'), axis=1), self.n_y)
	h = tf.concat([x,y,a], axis=1)
	_, _, z = dgm.samplePassGauss(h, self.qz_xya, self.n_hid, self.nonlinearity, self.bn, True, 'qz_xya')
	return z

    def compute_acc(self, x, y):
	y_, _ = self.predict(x)
	acc =  tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_,axis=1), tf.argmax(y, axis=1)), tf.float32))
	return acc 

    def initialize_networks(self):
    	""" Initialize all model networks """
	if self.x_dist == 'Gaussian':
      	    self.px_yz = dgm.initGaussNet(self.n_z+self.n_y, self.n_hid, self.n_x, 'px_yz_')
	elif self.x_dist == 'Bernoulli':
	    self.px_yz = dgm.initCatNet(self.n_z+self.n_y, self.n_hid, self.n_x, 'px_yz_')
	self.pa_xyz = dgm.initGaussNet(self.n_z+self.n_x+self.n_y, self.n_hid, self.n_a, 'pa_xyz_') 
    	self.qz_xya = dgm.initGaussNet(self.n_x+self.n_a+self.n_y, self.n_hid, self.n_z, 'qz_xya_')
    	self.qa_x = dgm.initGaussNet(self.n_x, self.n_hid, self.n_a, 'qa_x_')
    	self.qy_xa = dgm.initCatNet(self.n_x+self.n_a, self.n_hid, self.n_y, 'qy_xa_')

    def print_verbose(self, verbose, epoch, fd, sess):
	train_acc, test_acc, elbo_l, elbo_u = sess.run([self.train_acc, self.test_acc, self.elbo_l, self.elbo_u] ,fd)
	print("Epoch: {}: Labeled: {:5.3f}, Unlabeled: {:5.3f}, Training: {:5.3f}, Testing: {:5.3f}".format(epoch, elbo_l, elbo_u, train_acc, test_acc))	
	 
