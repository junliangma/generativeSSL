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
   
    def __init__(self, n_x, n_y, n_z=2, n_a=2, n_hid=[4], alpha=0.1, x_dist='Gaussian', nonlinearity=tf.nn.relu, batchnorm=False, mc_samples=1, l2_reg=1.0, ckpt=None):
	
	self.n_a = n_a        # auxiliary variable dimension
   	super(adgm, self).__init__(n_x, n_y, n_z, n_hid, x_dist, nonlinearity, batchnorm, mc_samples, alpha, l2_reg, ckpt)	

	""" TODO: add any general terms we want to have here """
	self.name = 'adgm'

    def build_model(self):
	""" Define model components and variables """
	self.create_placeholders()
	self.initialize_networks()
	## model variables and relations ##
	# inference #
	self.qa_mean, self.qa_lv, self.a_ = dgm.samplePassGauss(self.x, self.qa_x, self.n_hid, self.nonlinearity, self.bn, scope='qa_x', reuse=False)
	self.a_ = tf.reshape(self.a_, [-1, self.n_a])
	self.qy_in = tf.concat([self.x, self.a_], axis=-1)
	self.y_ = dgm.forwardPassCatLogits(self.qy_in, self.qy_xa, self.n_hid, self.nonlinearity, self.bn, scope='qy_xa', reuse=False)
	self.qz_in = tf.concat([self.x, self.y_, self.a_], axis=-1)
        self.qz_mean, self.qz_lv, self.z_ = dgm.samplePassGauss(self.qz_in, self.qz_xya, self.n_hid, self.nonlinearity, self.bn, scope='qz_xya', reuse=False) 	
	self.z_ = tf.reshape(self.z_, [-1, self.n_z])
	# generative #
	self.z_prior = tf.random_normal([tf.shape(self.y)[0], self.n_z])
	self.px_in = tf.concat([self.y, self.z_prior], axis=-1)
        if self.x_dist == 'Gaussian':
            self.px_mean, self.px_lv, self.x_ = dgm.samplePassGauss(self.px_in, self.px_yz, self.n_hid, self.nonlinearity, self.bn, scope='px_yz', reuse=False)
        elif self.x_dist == 'Bernoulli':
            self.x_ = dgm.forwardPassBernoulli(self.px_in, self.px_yz, self.n_hid, self.nonlinearity, self.bn, scope='px_yz', reuse=False)
	self.pa_in = tf.concat([self.x, self.y, self.z_], axis=-1)
	self.pa_mean, self.pa_lv, self.pa_ = dgm.samplePassGauss(self.pa_in, self.pa_xyz, self.n_hid, self.nonlinearity, self.bn, scope='pa_xyz', reuse=False)
	self.predictions = self.predict(self.x, training=True)

    def compute_loss(self):
	""" manipulate computed components and compute loss """
	elbo_l, qy_l = self.labeled_loss(self.x_l, self.y_l)
	self.elbo_l, self.qy_l = -tf.reduce_mean(elbo_l), -tf.reduce_mean(qy_l)
	self.elbo_u = -tf.reduce_mean(self.unlabeled_loss(self.x_u))
	weight_priors = self.l2_reg * self.weight_prior()/self.n_train
	return self.elbo_l + self.elbo_u - weight_priors

    def labeled_loss(self, x, y):
	""" Compute necessary terms for labeled loss (per data point) """
	qa_m, qa_lv, a = self.sample_a(x)
	qa_m, qa_lv = tf.tile(tf.expand_dims(qa_m,0), [self.mc_samples,1,1]), tf.tile(tf.expand_dims(qa_lv,0),[self.mc_samples,1,1])
	x_ = tf.tile(tf.expand_dims(x,0), [self.mc_samples, 1,1])
	y_ = tf.tile(tf.expand_dims(y,0), [self.mc_samples, 1,1])
	z_m, z_lv, z = self.sample_z(x_,y_,a)
	l_loss = self.lowerBound(x_, y_, z, z_m, z_lv, a, qa_m, qa_lv)  
	additional = tf.reduce_mean(self.qy_term(x_,y_,a) * self.alpha, axis=0)
	return l_loss + additional, additional

    def unlabeled_loss(self, x):
	""" Compute necessary terms for unlabeled loss (per data point) """
	### generate variables and shape them correctly ###
	qa_m, qa_lv, a = self.sample_a(x)
	qy_l = self.predict(x,a)
	x_r, a_r = tf.tile(x, [self.n_y,1]), tf.tile(a, [1,self.n_y,1])
	qa_mr, qa_lvr = tf.tile(tf.expand_dims(qa_m,0), [1,self.n_y,1]), tf.tile(tf.expand_dims(qa_lv,0), [1,self.n_y,1])
        y_u = tf.reshape(tf.tile(tf.eye(self.n_y), [1, tf.shape(self.x_u)[0]]), [-1, self.n_y])
	x_ = tf.tile(tf.expand_dims(x_r,0), [self.mc_samples, 1,1])
	y_ = tf.tile(tf.expand_dims(y_u,0), [self.mc_samples, 1,1])
	z_m, z_lv, z = self.sample_z(x_, y_, a_r)
        n_u = tf.shape(x)[0]
        lb_u = tf.transpose(tf.reshape(self.lowerBound(x_, y_, z, z_m, z_lv, a_r, qa_mr, qa_lvr), [self.n_y, n_u]))
        lb_u = tf.reduce_sum(qy_l * lb_u, axis=-1)
        qy_entropy = -tf.reduce_sum(qy_l * tf.log(qy_l + 1e-10), axis=-1)
        return lb_u + qy_entropy

    def lowerBound(self, x, y, z, z_m, z_lv, a, qa_m, qa_lv):
	""" Helper function for loss computations. Assumes each input is a rank(3) tensor """
	pa_in = tf.reshape(tf.concat([x, y, z], axis=-1), [-1, self.n_x + self.n_y + self.n_z])
	pa_m, pa_lv = dgm.forwardPassGauss(pa_in, self.pa_xyz, self.n_hid, self.nonlinearity, self.bn, scope='pa_xyz')
	pa_m, pa_lv = tf.reshape(pa_m, [self.mc_samples,-1,self.n_a]), tf.reshape(pa_lv, [self.mc_samples,-1,self.n_a])
	l_px = self.compute_logpx(x,y,z)
	l_py = dgm.multinoulliUniformLogDensity(y)
	l_pz = dgm.standardNormalLogDensity(z)
	l_pa = dgm.gaussianLogDensity(a, pa_m, pa_lv)
	l_qz = dgm.gaussianLogDensity(z, z_m, z_lv)
	l_qa = dgm.gaussianLogDensity(a, qa_m, qa_lv)
	return tf.reduce_mean(l_px + l_py + l_pz + l_pa - l_qz - l_qa, axis=0)	

    def qy_term(self, x, y, a):
	""" expected additional penalty under q(y|a,x) with samples from a"""
	qy_in = tf.reshape(tf.concat([x, a], axis=-1), [-1, self.n_x+self.n_a])
	y_ = tf.reshape(dgm.forwardPassCatLogits(qy_in, self.qy_xa, self.n_hid, self.nonlinearity, self.bn, True, 'qy_xa'), [self.mc_samples,-1,self.n_y])
        return dgm.multinoulliLogDensity(y,y_)
    
    def sample_a(self, x):
	""" return mc_samples samples from q(a|x)"""
        return dgm.samplePassGauss(x, self.qa_x, self.n_hid, self.nonlinearity, self.bn, mc_samps=self.mc_samples,  scope='qa_x')

    def sample_z(self, x, y, a):
	""" return mc_samples samples from q(z|x,y,a)"""
        l_qz_in = tf.reshape(tf.concat([x, y, a], axis=-1), [-1, self.n_x + self.n_y + self.n_z])
        z_m, z_lv, z =  dgm.samplePassGauss(l_qz_in, self.qz_xya, self.n_hid, self.nonlinearity, self.bn,  scope='qz_xya')
	z_m, z_lv = tf.reshape(z_m, [self.mc_samples,-1,self.n_z]), tf.reshape(z_lv, [self.mc_samples,-1,self.n_z])
	z = tf.reshape(z, [self.mc_samples,-1,self.n_z])
	return z_m, z_lv, z

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

    def predict(self, x, a=None, training=True):
	""" predict y for given x with q(y|x,a) """
	if a is None:
	    _, _, a = self.sample_a(x)
	qy_in = tf.reshape(tf.concat([tf.tile(tf.expand_dims(x,0), [self.mc_samples,1,1]), a], axis=-1), [-1, self.n_x+self.n_a])
	predictions = dgm.forwardPassCat(qy_in, self.qy_xa, self.n_hid, self.nonlinearity, self.bn, training, 'qy_xa')
	return tf.reduce_mean(tf.reshape(predictions, [self.mc_samples, -1, self.n_y]), axis=0) 

    def encode(self, x, y=None, n_iters=100):
	""" encode a new example into z-space (labeled or unlabeled) """
	pass	

    def compute_acc(self, x, y):
	y_  = self.predict(x)
	acc =  tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_,axis=1), tf.argmax(y, axis=1)), tf.float32))
	return acc 

    def initialize_networks(self):
    	""" Initialize all model networks """
	if self.x_dist == 'Gaussian':
      	    self.px_yz = dgm.initGaussNet(self.n_y+self.n_z, self.n_hid, self.n_x, 'px_yz_')
	elif self.x_dist == 'Bernoulli':
	    self.px_yz = dgm.initCatNet(self.n_y+self.n_z, self.n_hid, self.n_x, 'px_yz_')
	self.pa_xyz = dgm.initGaussNet(self.n_x+self.n_y+self.n_z, self.n_hid, self.n_a, 'pa_xyz_') 
    	self.qz_xya = dgm.initGaussNet(self.n_x+self.n_y+self.n_a, self.n_hid, self.n_z, 'qz_xya_')
    	self.qa_x = dgm.initGaussNet(self.n_x, self.n_hid, self.n_a, 'qa_x_')
    	self.qy_xa = dgm.initCatNet(self.n_x+self.n_a, self.n_hid, self.n_y, 'qy_xa_')

    def print_verbose1(self, epoch, fd, sess):
        total, elbo_l, elbo_u = sess.run([self.compute_loss(), self.elbo_l, self.elbo_u] ,fd)
        train_acc, test_acc = sess.run([self.train_acc, self.test_acc], fd)
        print("Epoch: {}: Total: {:5.3f}, Labeled: {:5.3f}, Unlabeled: {:5.3f}, Training: {:5.3f}, Testing: {:5.3f}".format(epoch, total, elbo_l, elbo_u, train_acc, test_acc)) 

    def print_verbose2(self, epoch, fd, sess):
        elbo_l, qy, weights = sess.run([self.elbo_l, self.qy_l, self.weight_prior()] ,fd)
        train_acc, test_acc = sess.run([self.train_acc, self.test_acc], fd)
        print("Epoch: {}: Labeled: {:5.3f}, Qy_l: {:5.3f}, Alpha: {:5.3f}, Weight reg: {:5.3f}"
                " Training: {:5.3f}, Testing: {:5.3f}".format(epoch, elbo_l, qy, self.alpha, weights/self.n_train, train_acc, test_acc))
