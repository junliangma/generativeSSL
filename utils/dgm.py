from __future__ import absolute_import
from __future__ import division 
from __future__ import print_function

import numpy as np

import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.contrib.distributions import RelaxedOneHotCategorical as Gumbel
import pdb

""" Module containing shared functions and structures for DGMS """

glorotNormal = xavier_initializer(uniform=False)
initNormal = tf.random_normal_initializer(stddev=1e-3)

############# Probability functions ##############

def gaussianLogDensity(inputs, mu, log_var):
    """ Gaussian log density """
    b_size = tf.cast(tf.shape(mu)[0], tf.float32)
    D = tf.cast(tf.shape(inputs)[-1], tf.float32)
    xc = inputs - mu
    return -0.5*(tf.reduce_sum((xc * xc) / tf.exp(log_var), axis=-1) + tf.reduce_sum(log_var, axis=-1) + D * tf.log(2.0*np.pi))

def standardNormalLogDensity(inputs):
    """ Standard normal log density """
    mu = tf.zeros_like(inputs)
    log_var = tf.log(tf.ones_like(inputs))
    return gaussianLogDensity(inputs, mu, log_var)

def bernoulliLogDensity(inputs, logits):
    """ Bernoulli log density """
    return -tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=inputs, logits=logits), axis=-1)

def multinoulliLogDensity(inputs, logits):
    """ Categorical log density """
    return -tf.nn.softmax_cross_entropy_with_logits(labels=inputs, logits=logits)

def multinoulliUniformLogDensity(inputs):
    """ Uniform Categorical log density """
    logits = tf.ones_like(inputs)
    return -tf.nn.softmax_cross_entropy_with_logits(labels=inputs, logits=logits)

def gumbelLogDensity(inputs, logits, temp):
    """ log density of a Gumbel distribution """
    dist = Gumbel(temperature=temp, logits=logits)
    return dist.log_prob(inputs)

def sampleNormal(mu, logvar, mc_samps):
    """ return a reparameterized sample from a Gaussian distribution """
    shape = tf.concat([tf.constant([mc_samps]), tf.shape(mu)], axis=-1)
    eps = tf.random_normal(shape, dtype=tf.float32)
    return mu + eps * tf.sqrt(tf.exp(logvar))

def sampleGumbel(logits, temp):
    """ return a reparameterized sample from a Gaussian distribution """
    shape = tf.shape(logits)
    U = tf.random_uniform(shape,minval=0,maxval=1)
    eps = -tf.log(-tf.log(U + 1e-10) + 1e-10)
    y = logits + eps
    return tf.nn.softmax( y / temp)

def standardNormalKL(mu, logvar):
    """ compute the KL divergence between a Gaussian and standard normal """
    return -0.5 * tf.reduce_sum(1 + logvar - mu**2 - tf.exp(logvar), axis=-1)

def gaussianKL(mu1, logvar1, mu2, logvar2):
    """ compute the KL divergence between two arbitrary Gaussians """
    return -0.5 * tf.reduce_sum(1 + logvar1 - logvar2 - tf.exp(logvar1)/tf.exp(logvar2) - ((mu1-mu2)**2)/tf.exp(logvar1), axis=-1)

############## Neural Network modules ##############

def initNetwork(n_in, n_hid, n_out, vname):
    weights = {}
    for layer, neurons in enumerate(n_hid):
        weight_name, bias_name = 'W'+str(layer), 'b'+str(layer)
        if layer == 0:
       	    weights[weight_name] = tf.get_variable(shape=[n_in, n_hid[layer]], name=vname+weight_name, initializer=glorotNormal)
    	else:
    	    weights[weight_name] = tf.get_variable(shape=[n_hid[layer-1], n_hid[layer]], name=vname+weight_name, initializer=glorotNormal)
    	weights[bias_name] = tf.get_variable(shape=[n_hid[layer]], name=vname+bias_name, initializer=initNormal)
    return weights

def initGaussNet(n_in, n_hid, n_out, vname):
    """ Initialize the weights of a network parameterizeing a Gaussian distribution"""
    weights = initNetwork(n_in, n_hid, n_out, vname)
    weights['Wmean'] = tf.get_variable(shape=[n_hid[-1], n_out], name=vname+'Wmean', initializer=initNormal)
    weights['bmean'] = tf.get_variable(shape=[n_out], name=vname+'bmean', initializer=initNormal)
    weights['Wvar'] = tf.get_variable(shape=[n_hid[-1], n_out], name=vname+'Wvar', initializer=initNormal)
    weights['bvar'] = tf.get_variable(shape=[n_out], name=vname+'bvar', initializer=initNormal)
    return weights

def initCatNet(n_in, n_hid, n_out, vname):
    """ Initialize the weights of a network parameterizeing a Gaussian distribution"""
    weights = initNetwork(n_in, n_hid, n_out, vname)
    weights['Wout'] = tf.get_variable(shape=[n_hid[-1], n_out], name=vname+'Wout', initializer=initNormal)
    weights['bout'] = tf.get_variable(shape=[n_out], name=vname+'bout', initializer=initNormal)
    return weights

def initGumbelNet(n_in, n_hid, n_out, vname, temp=1.0):
    """ Initialize the weights of a network parameterizeing a Gumbel-Softmax distribution"""
    weights = initCatNet(n_in, n_hid, n_out, vname)	
    return weights

def initTiedNetwork(nn1, n_hid, vname, n_type):
    """ Return a network tied by differences to an existing network nn1 """
    weights = {}
    for layer, neurons in enumerate(n_hid):
        weight_name, bias_name = 'W'+str(layer), 'b'+str(layer)
        eps_weight, eps_bias = 'epsW'+str(layer), 'epsb'+str(layer)
	weights[eps_weight] = tf.get_variable(shape=nn1[weight_name].shape, name=vname+eps_weight, initializer=initNormal)
	weights[eps_bias] = tf.get_variable(shape=nn1[bias_name].shape, name=vname+eps_bias, initializer=initNormal)
    	weights[weight_name] = tf.add(nn1[weight_name], weights[eps_weight], name=vname+weight_name) 
	weights[bias_name] = tf.add(nn1[bias_name], weights[eps_bias], name=vname+bias_name)
    if n_type=='Gauss':
	weights['eps_Wmean'] = tf.get_variable(shape=nn1['Wmean'].shape, name=vname+'eps_Wmean', initializer=initNormal)
	weights['eps_bmean'] = tf.get_variable(shape=nn1['bmean'].shape, name=vname+'eps_bmean', initializer=initNormal)
	weights['eps_Wvar'] = tf.get_variable(shape=nn1['Wvar'].shape, name=vname+'eps_Wvar', initializer=initNormal)
	weights['eps_bvar'] = tf.get_variable(shape=nn1['bvar'].shape, name=vname+'eps_bvar', initializer=initNormal)
        weights['Wmean'] = tf.add(nn1['Wmean'], weights['eps_Wmean'], name=vname+'Wmean')
        weights['bmean'] = tf.add(nn1['bmean'], weights['eps_bmean'], name=vname+'bmean')
        weights['Wvar'] = tf.add(nn1['Wvar'], weights['eps_Wvar'], name=vname+'Wvar')
        weights['bvar'] = tf.add(nn1['Wvar'], weights['eps_Wvar'], name=vname+'Wvar')
    elif n_type=='Categorical':
	weights['eps_Wout'] = tf.get_variable(shape=nn1['Wout'].shape, name=vname+'eps_Wout', initializer=initNormal)
	weights['eps_bout'] = tf.get_variable(shape=nn1['bout'].shape, name=vname+'eps_bout', initializer=initNormal)
        weights['Wout'] = tf.add(nn1['Wout'], weights['eps_Wout'], name=vname+'Wout')
        weights['bout'] = tf.add(nn1['bout'], weights['eps_bout'], name=vname+'bout')
    return weights
    

def forwardPass(x, weights, n_h, nonlinearity, bn, training, scope, reuse):
    h = x
    for layer, neurons in enumerate(n_h):
	weight_name, bias_name = 'W'+str(layer), 'b'+str(layer)
	h = tf.matmul(h, weights[weight_name]) + weights[bias_name]
	if bn:
	    name = scope+'_bn'+str(layer)
	    h = tf.layers.batch_normalization(h, training=training, name=name, reuse=reuse, momentum=0.99)
	h = nonlinearity(h)
    return h	

def forwardPassGauss(x, weights, n_h, nonlinearity, bn, training=True, scope='scope', reuse=True):
    """ Forward pass through the network with given weights - Gaussian output """
    h = forwardPass(x, weights, n_h, nonlinearity, bn, training, scope, reuse)
    mean = tf.matmul(h, weights['Wmean']) + weights['bmean']
    logVar = tf.matmul(h, weights['Wvar']) + weights['bvar']
    return mean, logVar

def samplePassGauss(x, weights, n_h, nonlinearity, bn, mc_samps=1, training=True, scope='scope', reuse=True):
    """ Forward pass through the network with given weights - Gaussian sampling """
    mean, logVar = forwardPassGauss(x, weights, n_h, nonlinearity, bn, training, scope, reuse)
    return mean, logVar, sampleNormal(mean, logVar, mc_samps)

def forwardPassCatLogits(x, weights, n_h, nonlinearity, bn, training=True, scope='scope', reuse=True):
    """ Forward pass through the network with weights as a dictionary """
    h = forwardPass(x, weights, n_h, nonlinearity, bn, training, scope, reuse) 
    logits = tf.matmul(h, weights['Wout']) + weights['bout']
    return logits

def forwardPassCat(x, weights, n_h, nonlinearity, bn=False, training=True, scope='scope', reuse=True):
    """ Forward pass through network with given weights - Categorical output """
    return tf.nn.softmax(forwardPassCatLogits(x, weights, n_h, nonlinearity, bn, training, scope, reuse))

def forwardPassBernoulli(x, weights, n_h, nonlinearity, bn=False, training=True, scope='scope', reuse=True):
    """ Forward pass through the network with given weights - Bernoulli output """
    return tf.nn.sigmoid(forwardPassCatLogits(x, weights, n_h, nonlinearity, bn, training, scope, reuse))

def forwardPassGumbel(x, weights, n_h, nonlinearity, bn=False, training=True, scope='scope', reuse=True):
    """ Forward pass through the network with given weights - Gumbel logits output """
    h = forwardPass(x, weights, n_h, nonlinearity, bn, training, scope, reuse) 
    logits = tf.matmul(h, weights['Wout']) + weights['bout']
    return logits

def samplePassGumbel(x, weights, n_h, nonlinearity, bn, temp, mc_samps=1, training=True, scope='scope', reuse=True):
    """ Forward pass through the network with given weights - return a Gumbel-softmax sample """
    logits = forwardPassCat(x, weights, n_h, nonlinearity, bn, training, scope, reuse)
    return logits, sampleGumbel(logits, temp)

