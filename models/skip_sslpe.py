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
Implementation of skip DGM: p(z) * p(a|z) * p(x|z,a) * p(y|x,z,a) 
Inference network: q(x,z,a|x) = q(a|x) * q(z|x,a) * q(y|x,z,a) 
"""

class skip_sslpe(model):
   
    def __init__(self, n_x, n_y, n_z=2, n_a=2, n_hid=[4], alpha=0.1, x_dist='Gaussian', nonlinearity=tf.nn.relu, batchnorm=False, mc_samples=1, l2_reg=1.0, ckpt=None):
	
	self.n_a = n_a        # auxiliary variable dimension
   	super(skip_sslpe, self).__init__(n_x, n_y, n_z, n_hid, x_dist, nonlinearity, batchnorm, mc_samples, alpha, l2_reg, ckpt)	

	""" TODO: add any general terms we want to have here """
	self.name = 'skip_sslpe'

    def build_model(self):
	""" Define model components and variables """
	self.create_placeholders()
	self.initialize_networks()
	## model variables and relations ##
	# inference #
	self.qa_mean, self.qa_lv, self.a_ = dgm.samplePassGauss(self.x, self.qa_x, self.n_hid, self.nonlinearity, self.bn, scope='qa_x', reuse=False)
	self.a_ = tf.reshape(self.a_, [-1, self.n_a])
	self.qz_in = tf.concat([self.x, self.a_], axis=-1)
	self.qz_mean, self.qz_lv, self.z_ = dgm.samplePassGauss(self.qz_in, self.qz_xa, self.n_hid, self.nonlinearity, self.bn, scope='qz_xa', reuse=False) 	
	self.z_ = tf.reshape(self.z_, [-1, self.n_z])
	self.qy_in = tf.concat([self.x, self.z_, self.a_], axis=-1)
        self.y_ = dgm.forwardPassCat(self.qy_in, self.qy_xza, self.n_hid, self.nonlinearity, self.bn, scope='qy_xza', reuse=False) 	
	# generative #
	self.z_prior = tf.random_normal([100, self.n_z])
	_, _, self.pa_ = dgm.samplePassGauss(self.z_prior, self.pa_z, self.n_hid, self.nonlinearity, self.bn, scope='pa_z', reuse=False)
	self.pa_ = tf.reshape(self.pa_, [-1, self.n_a])
	self.px_in = tf.concat([self.z_prior, self.pa_], axis=-1)
        if self.x_dist == 'Gaussian':
            self.px_mean, self.px_lv, self.x_ = dgm.samplePassGauss(self.px_in, self.px_za, self.n_hid, self.nonlinearity, self.bn, scope='px_za', reuse=False)
        elif self.x_dist == 'Bernoulli':
            self.x_ = dgm.forwardPassBernoulli(self.px_in, self.px_za, self.n_hid, self.nonlinearity, self.bn, scope='px_za', reuse=False)
	self.py_in = tf.concat([self.x_, self.z_prior, self.pa_], axis=-1)
	self.py_ = dgm.forwardPassCat(self.py_in, self.py_xza, self.n_hid, self.nonlinearity, self.bn, scope='py_xza', reuse=False)
	self.predictions = self.predict(self.x)
	
    def compute_loss(self):
	""" manipulate computed components and compute loss """
	self.elbo_l = tf.reduce_mean(self.labeled_loss(self.x_l, self.y_l))
	self.qy_ll = tf.reduce_mean(self.qy_loss(self.x_l, self.y_l))
	self.elbo_u = tf.reduce_mean(self.unlabeled_loss(self.x_u))
	weight_priors = self.l2_reg * self.weight_prior()/self.n_train
	return -(self.elbo_l + self.elbo_u + self.alpha * self.qy_ll +  weight_priors)

    def labeled_loss(self, x, y):
	""" compute necessary terms for labeled loss (per data point) """
	qa_m, qa_lv, a = self.sample_a(x)
	z_m, z_lv, z = self.sample_z(x,a)
	x_ = tf.tile(tf.expand_dims(x,0), [self.mc_samples, 1,1])
	y_ = tf.tile(tf.expand_dims(y,0), [self.mc_samples, 1,1])
	return self.lowerBound(x_, y_, z, z_m, z_lv, a, qa_m, qa_lv)  

    def unlabeled_loss(self, x):
	""" TODO: compute necessary terms for unlabeled loss (per data point) """
	### generate variables and shape them correctly ###
	qa_m, qa_lv, a = self.sample_a(x)
	z_m, z_lv, z = self.sample_z(x,a)
	qy_l = self.predict(x,z,a)
	x_r, a_r, z_r = tf.tile(x, [self.n_y,1]), tf.tile(a, [1,self.n_y,1]), tf.tile(z, [1,self.n_y,1])
	qa_mr, qa_lvr = tf.tile(tf.expand_dims(qa_m,0), [1,self.n_y,1]), tf.tile(tf.expand_dims(qa_lv,0), [1,self.n_y,1])
	z_mr, z_lvr = tf.tile(z_m, [1,self.n_y,1]), tf.tile(tf.expand_dims(qa_lv,0), [1,self.n_y,1])
        y_u = tf.reshape(tf.tile(tf.eye(self.n_y), [1, tf.shape(self.x_u)[0]]), [-1, self.n_y])
	x_ = tf.tile(tf.expand_dims(x_r,0), [self.mc_samples, 1,1])
	y_ = tf.tile(tf.expand_dims(y_u,0), [self.mc_samples, 1,1])
        n_u = tf.shape(x)[0]
        lb_u = tf.transpose(tf.reshape(self.lowerBound(x_, y_, z, z_m, z_lv, a_r, qa_mr, qa_lvr), [self.n_y, n_u]))
        lb_u = tf.reduce_sum(qy_l * lb_u, axis=-1)
        qy_entropy = -tf.reduce_sum(qy_l * tf.log(qy_l + 1e-10), axis=-1)
        return lb_u + qy_entropy

    def lowerBound(self, x, y, z, z_m, z_lv, a, qa_m, qa_lv):
	""" Helper function for loss computations. Assumes each input is a rank(3) tensor """
	pa_in = tf.reshape(z, [-1, self.n_z])
	pa_m, pa_lv = dgm.forwardPassGauss(pa_in, self.pa_z, self.n_hid, self.nonlinearity, self.bn, scope='pa_z')
	pa_m, pa_lv = tf.reshape(pa_m, [self.mc_samples, -1, self.n_a]), tf.reshape(pa_lv, [self.mc_samples, -1, self.n_a])
	l_px = self.compute_logpx(x,z,a)
	l_py = self.compute_logpy(y,x,z,a)
	l_pz = dgm.standardNormalLogDensity(z)
	l_pa = dgm.gaussianLogDensity(a, pa_m, pa_lv)
	l_qz = dgm.gaussianLogDensity(z, z_m, z_lv)
	l_qa = dgm.gaussianLogDensity(a, qa_m, qa_lv)
	return tf.reduce_mean(l_px + l_py + l_pz + l_pa - l_qz - l_qa, axis=0)	

    def qy_loss(self, x, y):
	""" expected additional penalty under q(y|a,x) with samples from a"""
	_, _, a = self.sample_a(x)
	x_ = tf.tile(tf.expand_dims(x,0), [self.mc_samples,1,1])
	qz_in = tf.reshape(tf.concat([x_, a], axis=-1), [-1, self.n_x+self.n_a])
	_, _, z = dgm.samplePassGauss(qz_in, self.qz_xa, self.n_hid, self.nonlinearity, self.bn, reuse=True, scope='qz_xa')
	z = tf.reshape(tf.expand_dims(z,0), [self.mc_samples, -1, self.n_z])
	qy_in = tf.reshape(tf.concat([x_,z,a], axis=-1), [-1,self.n_x+self.n_z+self.n_a])
	y_ = dgm.forwardPassCatLogits(qy_in, self.qy_xza, self.n_hid, self.nonlinearity, self.bn, reuse=True, scope='qy_xza')
	y_ = tf.reshape(tf.expand_dims(y_,0), [self.mc_samples,-1,self.n_y])
        return tf.reduce_mean(dgm.multinoulliLogDensity(tf.tile(tf.expand_dims(y,0),[self.mc_samples,1,1]), y_),axis=0)
    
    def sample_a(self, x):
	""" return mc_samples samples from q(a|x)"""
        return dgm.samplePassGauss(x, self.qa_x, self.n_hid, self.nonlinearity, self.bn, mc_samps=self.mc_samples,  scope='qa_x')

    def sample_z(self, x, a):
	""" return mc_samples samples from q(z|x,y,a)"""
	x_ = tf.tile(tf.expand_dims(x,0), [self.mc_samples,1,1])
        l_qz_in = tf.reshape(tf.concat([x_, a], axis=-1), [-1, self.n_x + self.n_z])
        z_m, z_lv, z =  dgm.samplePassGauss(l_qz_in, self.qz_xa, self.n_hid, self.nonlinearity, self.bn,  scope='qz_xa')
	z_m, z_lv = tf.reshape(z_m, [self.mc_samples,-1,self.n_z]), tf.reshape(z_lv, [self.mc_samples,-1,self.n_z])
	z = tf.reshape(z, [self.mc_samples,-1,self.n_z])
	return z_m, z_lv, z

    def compute_logpx(self, x, z, a):
        px_in = tf.reshape(tf.concat([z,a],axis=-1), [-1, self.n_z+self.n_a])
        if self.x_dist == 'Gaussian':
            mean, log_var = dgm.forwardPassGauss(px_in, self.px_za, self.n_hid, self.nonlinearity, self.bn, scope='px_za')
            mean, log_var = tf.reshape(mean, [self.mc_samples, -1, self.n_x]),  tf.reshape(log_var, [self.mc_samples, -1, self.n_x])
            return dgm.gaussianLogDensity(x, mean, log_var)
        elif self.x_dist == 'Bernoulli':
            logits = dgm.forwardPassCatLogits(px_in, self.px_za, self.n_hid, self.nonlinearity, self.bn, scope='px_za')
            logits = tf.reshape(logits, [self.mc_samples, -1, self.n_x])
            return dgm.bernoulliLogDensity(x, logits)

    def compute_logpy(self, y, x, z, a):
        """ compute the log density of y under p(y|x,z,a)"""
	pdb.set_trace()
        py_in = tf.reshape(tf.concat([x,z,a], axis=-1), [-1, self.n_x+self.n_z+self.n_a])
        y_ = dgm.forwardPassCatLogits(py_in, self.py_xza, self.n_hid, self.nonlinearity, self.bn, scope='py_xza')
        y_ = tf.reshape(y_, [self.mc_samples, -1, self.n_y])
        return dgm.multinoulliLogDensity(y, y_)

    def predict(self, x, z=None, a=None):
	""" predict y for given x with q(a|x) -> q(z|x,a) -> p(y|x,z,a) """
	if a is None:
	    _, _, a = self.sample_a(x)
	if z is None:
	    _, _, z = self.sample_z(x,a)
	py_in = tf.reshape(tf.concat([tf.tile(tf.expand_dims(x,0), [self.mc_samples,1,1]), z, a], axis=-1), [-1, self.n_x+self.n_z+self.n_a])
        y_ = dgm.forwardPassCat(py_in, self.py_xza, self.n_hid, self.nonlinearity, self.bn, scope='py_xza')
	return tf.reduce_mean(tf.reshape(y_, [self.mc_samples,-1,self.n_y]), axis=0)

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
	y_  = self.predict(x)
	acc =  tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_,axis=1), tf.argmax(y, axis=1)), tf.float32))
	return acc 

    def initialize_networks(self):
    	""" Initialize all model networks """
	if self.x_dist == 'Gaussian':
      	    self.px_za = dgm.initGaussNet(self.n_z+self.n_a, self.n_hid, self.n_x, 'px_za_')
	elif self.x_dist == 'Bernoulli':
	    self.px_za = dgm.initCatNet(self.n_z+self.n_a, self.n_hid, self.n_x, 'px_za_')
	self.pa_z = dgm.initGaussNet(self.n_z, self.n_hid, self.n_a, 'pa_z_') 
    	self.qz_xa = dgm.initGaussNet(self.n_x+self.n_a, self.n_hid, self.n_z, 'qz_xa_')
    	self.qa_x = dgm.initGaussNet(self.n_x, self.n_hid, self.n_a, 'qa_x_')
    	self.qy_xza = dgm.initCatNet(self.n_x+self.n_z+self.n_a, self.n_hid, self.n_y, 'qy_xza_')
    	self.py_xza = dgm.initCatNet(self.n_x+self.n_z+self.n_a, self.n_hid, self.n_y, 'py_xza_')

    def print_verbose1(self, epoch, fd, sess):
        total, elbo_l, elbo_u = sess.run([self.compute_loss(), self.elbo_l, self.elbo_u] ,fd)
        train_acc, test_acc = sess.run([self.train_acc, self.test_acc], fd)
        print("Epoch: {}: Total: {:5.3f}, Labeled: {:5.3f}, Unlabeled: {:5.3f}, Training: {:5.3f}, Testing: {:5.3f}".format(epoch, total, elbo_l, elbo_u, train_acc, test_acc)) 
