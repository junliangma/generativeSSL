from __future__ import absolute_import
from __future__ import division 
from __future__ import print_function

from models.model import model

import sys, os, pdb

import numpy as np
import utils.dgm as dgm 
import utils.bnn as bnn 

import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector


""" 
Implementation of blended skip DGMs from Maaloe et al. (2016) with Bayesian discriminator: [p(z) * p(y) * p(a|z,y) * p(x|,y,z,a)] * p(y|x,a)
Inference network: q(a,z,y|x) = q(a|x) * q(y|a,x) * q(z|a,y,x) 
"""

class b_sblended(model):
   
    def __init__(self, n_x, n_y, n_z=2, n_a=2, n_hid=[4], alpha=0.1, beta=None, initVar=-5., x_dist='Gaussian', nonlinearity=tf.nn.relu, batchnorm=False, mc_samples=1, l2_reg=1.0, ckpt=None):
	
	if beta is None:
	    self.beta = tf.get_variable(name='beta', dtype=tf.float32, initializer=tf.constant(0.1))
	else:
	    self.beta = beta
	self.sigma = (self.beta / (1-self.beta))**2
	self.reg_term = tf.placeholder(tf.float32, shape=[], name='reg_term')
	self.n_a = n_a        # auxiliary variable dimension
	self.initVar = initVar
   	super(b_sblended, self).__init__(n_x, n_y, n_z, n_hid, x_dist, nonlinearity, batchnorm, mc_samples, alpha, l2_reg, ckpt)	

	self.name = 'bayes_sblended'

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
	self.pa_in = tf.concat([self.y, self.z_], axis=-1)
	self.pa_mean, self.pa_lv, self.pa_ = dgm.samplePassGauss(self.pa_in, self.pa_yz, self.n_hid, self.nonlinearity, self.bn, scope='pa_yz', reuse=False)
	self.pa_ = tf.reshape(self.pa_, [-1, self.n_a])	
	self.px_in = tf.concat([self.y, self.z_prior, self.pa_], axis=-1)
        if self.x_dist == 'Gaussian':
            self.px_mean, self.px_lv, self.x_ = dgm.samplePassGauss(self.px_in, self.px_yza, self.n_hid, self.nonlinearity, self.bn, scope='px_yza', reuse=False)
        elif self.x_dist == 'Bernoulli':
            self.x_ = dgm.forwardPassBernoulli(self.px_in, self.px_yza, self.n_hid, self.nonlinearity, self.bn, scope='px_yza', reuse=False)
	self.da_mean, self.da_lv, self.da_ = dgm.samplePassGauss(self.x, self.pa_x, self.n_hid, self.nonlinearity, self.bn, scope='pa_x', reuse=False)
	self.da_ = tf.reshape(self.a_, [-1, self.n_a])
	self.dy_in = tf.concat([self.x, self.da_], axis=-1)
	self.W = bnn.sampleCatBNN(self.py_xa, self.n_hid)
	self.dy_ = dgm.forwardPassCatLogits(self.dy_in, self.W, self.n_hid, self.nonlinearity, self.bn, scope='py_xa', reuse=False)
	self.predictions = self.predict(self.x, training=False, n_samps=25)
	
    def compute_loss(self):
	""" manipulate computed components and compute loss """
	self.elbo_l = tf.reduce_mean(self.labeled_loss(self.x_l, self.y_l))
	self.elbo_u = tf.reduce_mean(self.unlabeled_loss(self.x_u))
	self.weight_priors = self.compute_prior()
	kl_term = self.klCatBNN(self.py_xa, self.n_hid)/self.reg_term 
	return -(self.elbo_l + self.elbo_u + self.weight_priors - kl_term)
	
    def labeled_loss(self, x, y):
	""" Compute necessary terms for labeled loss (per data point) """
	qa_m, qa_lv, a = self.sample_qa(x)
	qa_m, qa_lv = tf.tile(tf.expand_dims(qa_m,0), [self.mc_samples,1,1]), tf.tile(tf.expand_dims(qa_lv,0),[self.mc_samples,1,1])
	x_ = tf.tile(tf.expand_dims(x,0), [self.mc_samples, 1,1])
	y_ = tf.tile(tf.expand_dims(y,0), [self.mc_samples, 1,1])
	z_m, z_lv, z = self.sample_z(x_,y_,a)
	g_term = self.lowerBound(x_, y_, z, z_m, z_lv, a, qa_m, qa_lv) 
	d_term = self.discriminatorTerm(x,y)
	return g_term + self.alpha * d_term 

    def unlabeled_loss(self, x):
	""" Compute necessary terms for unlabeled loss (per data point) """
	### generate variables and shape them correctly ###
	qa_m, qa_lv, a = self.sample_qa(x)
	qy_l = self.predictq(x,a)
	x_r, a_r = tf.tile(x, [self.n_y,1]), tf.tile(a, [1,self.n_y,1])
 	qa_mr, qa_lvr = tf.tile(tf.expand_dims(qa_m,0), [1,self.n_y,1]), tf.tile(tf.expand_dims(qa_lv,0), [1,self.n_y,1])
        y_u = tf.reshape(tf.tile(tf.eye(self.n_y), [1, tf.shape(x)[0]]), [-1, self.n_y])
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
	pa_in = tf.reshape(tf.concat([y, z], axis=-1), [-1,self.n_y + self.n_z])
	pa_m, pa_lv = dgm.forwardPassGauss(pa_in, self.pa_yz, self.n_hid, self.nonlinearity, self.bn, scope='pa_yz')
	pa_m, pa_lv = tf.reshape(pa_m, [self.mc_samples,-1,self.n_a]), tf.reshape(pa_lv, [self.mc_samples,-1,self.n_a])
	l_px = self.compute_logpx(x,y,z,a)
	l_py = dgm.multinoulliUniformLogDensity(y)
	l_pz = dgm.standardNormalLogDensity(z)
	l_pa = dgm.gaussianLogDensity(a, pa_m, pa_lv)
	l_qz = dgm.gaussianLogDensity(z, z_m, z_lv)
	l_qa = dgm.gaussianLogDensity(a, qa_m, qa_lv)
	return tf.reduce_mean(l_px + l_py + l_pz + l_pa - l_qz - l_qa, axis=0)	

    def discriminatorTerm(self, x, y, n_samps=5):
	""" compute log p(y|x,a) with a~p(a|x) """
	_, _, a = self.sample_pa(x)
	x_ = tf.tile(tf.expand_dims(x,0),[self.mc_samples,1,1])
	py_in = tf.reshape(tf.concat([x_,a],-1), [-1,self.n_x+self.n_a])
	self.W = bnn.sampleCatBNN(self.py_xa, self.n_hid)
	preds = dgm.forwardPassCatLogits(py_in, self.W, self.n_hid, self.nonlinearity, self.bn, scope='py_xa')
	preds = tf.expand_dims(tf.reduce_mean(tf.reshape(preds,[self.mc_samples,-1,self.n_y]),0),axis=0) 
	for sample in range(n_samps-1):
  	    self.W = bnn.sampleCatBNN(self.py_xa, self.n_hid)
	    preds_new = dgm.forwardPassCatLogits(py_in, self.W, self.n_hid, self.nonlinearity, self.bn, scope='py_xa')
   	    logits = tf.reduce_mean(tf.reshape(preds_new,[self.mc_samples,-1,self.n_y]), axis=0)
	    preds = tf.concat([preds, tf.expand_dims(logits,0)], axis=0)
	return dgm.multinoulliLogDensity(y, tf.reduce_mean(preds,0))
 
    def sample_qa(self, x):
	""" return mc_samples samples from q(a|x)"""
        return dgm.samplePassGauss(x, self.qa_x, self.n_hid, self.nonlinearity, self.bn, mc_samps=self.mc_samples,  scope='qa_x')

    def sample_pa(self, x):
	""" return mc_samples samples from q(a|x)"""
        return dgm.samplePassGauss(x, self.pa_x, self.n_hid, self.nonlinearity, self.bn, mc_samps=self.mc_samples,  scope='pa_x')

    def sample_z(self, x, y, a):
	""" return mc_samples samples from q(z|x,y,a)"""
        l_qz_in = tf.reshape(tf.concat([x, y, a], axis=-1), [-1, self.n_x + self.n_y + self.n_z])
        z_m, z_lv, z =  dgm.samplePassGauss(l_qz_in, self.qz_xya, self.n_hid, self.nonlinearity, self.bn,  scope='qz_xya')
	z_m, z_lv = tf.reshape(z_m, [self.mc_samples,-1,self.n_z]), tf.reshape(z_lv, [self.mc_samples,-1,self.n_z])
	z = tf.reshape(z, [self.mc_samples,-1,self.n_z])
	return z_m, z_lv, z

    def compute_logpx(self, x, y, z, a):
	""" compute the log density of x under p(x|y,z,a) """
        px_in = tf.reshape(tf.concat([y,z,a], axis=-1), [-1, self.n_y + self.n_z+ self.n_a])
        if self.x_dist == 'Gaussian':
            mean, logVar = dgm.forwardPassGauss(px_in, self.px_yza, self.n_hid, self.nonlinearity, self.bn, scope='px_yza')
            mean, logVar = tf.reshape(mean, [self.mc_samples, -1, self.n_x]),  tf.reshape(logVar, [self.mc_samples, -1, self.n_x])
            return dgm.gaussianLogDensity(x, mean, logVar)
        elif self.x_dist == 'Bernoulli':
            logits = dgm.forwardPassCatLogits(px_in, self.px_yza, self.n_hid, self.nonlinearity, self.bn, scope='px_yza')
            logits = tf.reshape(logits, [self.mc_samples, -1, self.n_x])
            return dgm.bernoulliLogDensity(x, logits)

    def compute_prior(self):
        """ compute the log prior term """
	weights = [V for V in tf.trainable_variables() if 'py_xa' not in V.name]
        weight_term = np.sum([tf.reduce_sum(dgm.standardNormalLogDensity(w)) for w in weights])
        return  (self.l2_reg * weight_term) / self.reg_term

    def predict(self, x, a=None, training=True, n_samps=5):
	""" predict y for given x with q(y|x,a) """
	if a is None:
	    _, _, a = self.sample_pa(x)
	py_in = tf.reshape(tf.concat([tf.tile(tf.expand_dims(x,0), [self.mc_samples,1,1]), a], axis=-1), [-1, self.n_x+self.n_a])
	self.W = bnn.sampleCatBNN(self.py_xa, self.n_hid)
	preds = dgm.forwardPassCat(py_in, self.W, self.n_hid, self.nonlinearity, self.bn, scope='py_xa')
	preds = tf.expand_dims(tf.reduce_mean(tf.reshape(preds,[self.mc_samples,-1,self.n_y]),0),axis=0) 
	for sample in range(n_samps-1):
  	    self.W = bnn.sampleCatBNN(self.py_xa, self.n_hid)
	    preds_new = dgm.forwardPassCat(py_in, self.W, self.n_hid, self.nonlinearity, self.bn, scope='py_xa')
   	    logits = tf.reduce_mean(tf.reshape(preds_new,[self.mc_samples,-1,self.n_y]), axis=0)
	    preds = tf.concat([preds, tf.expand_dims(logits,0)],0)
	return tf.reduce_mean(preds,0)

    def predictq(self, x, a=None, training=True):
	""" predict y for given x with q(y|x,a) """
	if a is None:
	    _, _, a = self.sample_qa(x)
	qy_in = tf.reshape(tf.concat([tf.tile(tf.expand_dims(x,0), [self.mc_samples,1,1]), a], axis=-1), [-1, self.n_x+self.n_a])
	predictions = dgm.forwardPassCat(qy_in, self.qy_xa, self.n_hid, self.nonlinearity, self.bn, training, 'qy_xa')
	return tf.reduce_mean(tf.reshape(predictions, [self.mc_samples, -1, self.n_y]), axis=0) 

    def encode(self, x, y=None, n_iters=100):
	""" TODO: encode a new example into z-space (labeled or unlabeled) """
	pass

    def compute_acc(self, x, y):
	y_  = self.predict(x, training=False)
	acc =  tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_,axis=1), tf.argmax(y, axis=1)), tf.float32))
	return acc 

    def compute_accq(self, x, y):
	y_  = self.predictq(x)
	acc =  tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_,axis=1), tf.argmax(y, axis=1)), tf.float32))
	return acc 

    def initialize_networks(self):
    	""" Initialize all model networks """
	if self.x_dist == 'Gaussian':
      	    self.px_yza = dgm.initGaussNet(self.n_y+self.n_z+self.n_a, self.n_hid, self.n_x, 'px_yza_')
	elif self.x_dist == 'Bernoulli':
	    self.px_yza = dgm.initCatNet(self.n_y+self.n_z+self.n_a, self.n_hid, self.n_x, 'px_yza_')
	self.pa_yz = dgm.initGaussNet(self.n_y+self.n_z, self.n_hid, self.n_a, 'pa_yz_') 
    	self.qz_xya = dgm.initGaussNet(self.n_x+self.n_y+self.n_a, self.n_hid, self.n_z, 'qz_xya_')
    	self.qa_x = dgm.initGaussNet(self.n_x, self.n_hid, self.n_a, 'qa_x_')
    	self.pa_x = dgm.initTiedNetwork(self.qa_x, self.n_hid, 'pa_x_', 'Gauss')
    	self.qy_xa = dgm.initCatNet(self.n_x+self.n_a, self.n_hid, self.n_y, 'qy_xa_')
    	self.py_xa = bnn.initTiedBNN(self.qy_xa, self.n_hid, self.initVar, 'py_xa_', 'Categorical')

    def klBNN(self, q, n_hid):
        """ MC approximation KL(q||p) for reparameterized network"""
        kl = 0
        for layer, neurons in enumerate(n_hid):
            w_qMean, b_qMean = q['epsW'+str(layer)], tf.expand_dims(q['epsb'+str(layer)],1)
            w_qLv, b_qLv = q['W'+str(layer)+'_logvar'], tf.expand_dims(q['b'+str(layer)+'_logvar'],1)
            w_pMean, b_pMean = tf.zeros_like(w_qMean), tf.zeros_like(b_qMean)
            w_pLv, b_pLv = tf.fill(w_qLv.shape, tf.log(self.sigma**2)), tf.fill(b_qLv.shape, tf.log(self.sigma**2))
            kl += tf.reduce_sum(dgm.gaussianKL(w_qMean, w_qLv, w_pMean, w_pLv)) + tf.reduce_sum(dgm.gaussianKL(b_qMean, b_qLv, b_pMean, b_pLv))
        return kl

    def klCatBNN(self, q, n_hid):
        """ compute exact KL(q||p) with standard normal p(w) for reparameterized categorical BNN """
        kl = self.klBNN(q, n_hid)
        w_qMean, b_qMean = q['eps_Wout'], tf.expand_dims(q['eps_bout'],1)
        w_qLv, b_qLv = q['Wout_logvar'], tf.expand_dims(q['bout_logvar'],1)
        w_pMean, b_pMean = tf.zeros_like(w_qMean), tf.zeros_like(b_qMean)
        w_pLv, b_pLv = tf.fill(w_qLv.shape, tf.log(self.sigma**2)), tf.fill(b_qLv.shape, tf.log(self.sigma**2))
        kl += tf.reduce_sum(dgm.gaussianKL(w_qMean, w_qLv, w_pMean, w_pLv)) + tf.reduce_sum(dgm.gaussianKL(b_qMean, b_qLv, b_pMean, b_pLv))
        return kl

    def training_fd(self, x_l, y_l, x_u):
        return {self.x_l: x_l, self.y_l: y_l, self.x_u: x_u, self.x: x_l, self.y: y_l, self.reg_term:self.n_train}

    def _printing_feed_dict(self, Data, x_l, x_u, y, eval_samps, binarize):
        fd = super(b_sblended,self)._printing_feed_dict(Data, x_l, x_u, y, eval_samps, binarize)
        fd[self.reg_term] = self.n_train
        return fd

    def print_verbose1(self, epoch, fd, sess):
        total, elbo_l, elbo_u, qy = sess.run([self.compute_loss(), self.elbo_l, self.elbo_u, self.qy_l] ,fd)
        train_acc, test_acc = sess.run([self.train_acc, self.test_acc], fd)
        print("Epoch: {}: Total: {:5.3f}, Labeled: {:5.3f}, Unlabeled: {:5.3f}, Qy_l: {:5.3f}"
		" Training: {:5.3f}, Testing: {:5.3f}".format(epoch, total, elbo_l, elbo_u, qy, train_acc, test_acc))

    def print_verbose2(self, epoch, fd, sess):
        total, elbo_l, elbo_u = sess.run([self.compute_loss(), self.elbo_l, self.elbo_u] ,fd)
        train_acc, test_acc = sess.run([self.train_acc, self.test_acc], fd)
        trainq = sess.run(self.compute_accq(self.x_train, self.y_train), fd)
        testq = sess.run(self.compute_accq(self.x_test, self.y_test), fd)
        print("Epoch {}: Total: {:5.3f}, Labeled: {:5.3f}, Unlabeled: {:5.3f}, Training (p): {:5.3f}, Testing (p): {:5.3f} "
                "Training (q): {:5.3f}, Testing (q): {:5.3f}".format(epoch, total, elbo_l, elbo_u, train_acc, test_acc, trainq, testq)) 
