from __future__ import absolute_import
from __future__ import division 
from __future__ import print_function

import numpy as np

import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

import pdb

""" Module containing shared functions and structures for DGMS """

def _forward_pass_Gauss(x, weights, n_h, nonlinearity, bn, training):
    """ Forward pass through the network with given weights - Gaussian output """
    h = x
    for layer, neurons in enumerate(n_h):
	weight_name, bias_name = 'W'+str(layer), 'b'+str(layer)
	h = tf.matmul(h, weights[weight_name]) + weights[bias_name]
	if bn:
	    h = batch_norm_wrapper(h, weights, layer, training)
	h = nonlinearity(h)
    mean = tf.matmul(h, weights['Wmean']) + weights['bmean']
    log_var = tf.matmul(h, weights['Wvar']) + weights['bvar']
    return mean, log_var

def _forward_pass_Cat_logits(x, weights, n_h, nonlinearity, bn, training):
    """ Forward pass through the network with weights as a dictionary """
    h = x
    for layer, neurons in enumerate(n_h):
    	weight_name, bias_name = 'W'+str(layer), 'b'+str(layer)
	h = tf.matmul(h, weights[weight_name]) + weights[bias_name]
	if bn:
	    h = batch_norm_wrapper(h, weights, layer, training)
        h = nonlinearity(h)
    logits = tf.matmul(h, weights['Wout']) + weights['bout']
    return logits

def _forward_pass_Cat(x, weights, n_h, nonlinearity, bn=False, training=True):
    """ Forward pass through network with given weights - Categorical output """
    return tf.nn.softmax(_forward_pass_Cat_logits(x, weights, n_h, nonlinearity, bn, training))


def _forward_pass_Bernoulli(x, weights, n_h, nonlinearity, bn=False, training=True):
    """ Forward pass through the network with given weights - Bernoulli output """
    return tf.nn.sigmoid(_forward_pass_Cat_logits(x, weights, n_h, nonlinearity, bn, training))


def _gauss_kl(mean, log_var):
    return -0.5 * tf.reduce_sum(1 + log_var - mean**2 - tf.exp(log_var), axis=1)


def _gauss_logp(x, mu, log_var):
        b_size = tf.cast(tf.shape(mu)[0], tf.float32)
	D = tf.cast(tf.shape(x)[1], tf.float32)
        xc = x - mu
        return -0.5*(tf.reduce_sum((xc * xc) / tf.exp(log_var), axis=1) + tf.reduce_sum(log_var, axis=1) + D * tf.log(2.0*np.pi))


def _init_Gauss_net(n_in, architecture, n_out, vname, bn=False):
    """ Initialize the weights of a network parameterizeing a Gaussian distribution"""
    weights = {}
    for i, neurons in enumerate(architecture):
        weight_name, bias_name = 'W'+str(i), 'b'+str(i)
        if i == 0:
       	    weights[weight_name] = tf.Variable(xavier_initializer(n_in, architecture[i]), name=vname+weight_name)
    	else:
    	    weights[weight_name] = tf.Variable(xavier_initializer(architecture[i-1], architecture[i]), name=vname+weight_name)
	if bn:
	    scale, beta, mean, var = 'scale'+str(i), 'beta'+str(i), 'mean'+str(i), 'var'+str(i)
	    weights[scale] = tf.Variable(tf.ones(architecture[i]), name=vname+scale) 
	    weights[beta] = tf.Variable(tf.zeros(architecture[i]), name=vname+beta) 
	    weights[mean] = tf.Variable(tf.zeros(architecture[i]), name=vname+mean, trainable=False)
	    weights[var] = tf.Variable(tf.ones(architecture[i]), name=vname+var, trainable=False) 
    	weights[bias_name] = tf.Variable(tf.zeros(architecture[i]) + 1e-1, name=vname+bias_name)
    weights['Wmean'] = tf.Variable(xavier_initializer(architecture[-1], n_out), name=vname+'Wmean')
    weights['bmean'] = tf.Variable(tf.zeros(n_out) + 1e-1, name=vname+'bmean')
    weights['Wvar'] = tf.Variable(xavier_initializer(architecture[-1], n_out), name=vname+'Wvar')
    weights['bvar'] = tf.Variable(tf.zeros(n_out) + 1e-1, name=vname+'bvar')
    return weights


def _init_Cat_net(n_in, architecture, n_out, vname, bn=False):
    """ Initialize the weights of a network with batch normalization parameterizeing a Categorical distribution """
    weights = {}
    for i, neurons in enumerate(architecture):
        weight_name, bias_name = 'W'+str(i), 'b'+str(i)
        if i == 0:
            weights[weight_name] = tf.Variable(xavier_initializer(n_in, architecture[i]), name=vname+weight_name)
    	else:
    	    weights[weight_name] = tf.Variable(xavier_initializer(architecture[i-1], architecture[i]), name=vname+weight_name)
	if bn:
	    scale, beta, mean, var = 'scale'+str(i), 'beta'+str(i), 'mean'+str(i), 'var'+str(i)
	    weights[scale] = tf.Variable(tf.ones(architecture[i]), name=vname+scale) 
	    weights[beta] = tf.Variable(tf.zeros(architecture[i]), name=vname+beta) 
	    weights[mean] = tf.Variable(tf.zeros(architecture[i]), name=vname+mean, trainable=False) 
	    weights[var] = tf.Variable(tf.ones(architecture[i]), name=vname+var, trainable=False) 
    	weights[bias_name] = tf.Variable(tf.zeros(architecture[i]) + 1e-1, name=vname+bias_name)
    weights['Wout'] = tf.Variable(xavier_initializer(architecture[-1], n_out), name=vname+'Wout')
    weights['bout'] = tf.Variable(tf.zeros(n_out) + 1e-1, name=vname+'bout')
    return weights
	


def _init_Cat_bnn(n_in, architecture, n_out, vname, initVar=-5):
    weights = {}
    for i, neurons in enumerate(architecture):
        weight_mean, bias_mean = 'W'+str(i)+'_mean', 'b'+str(i)+'_mean'
        weight_logvar, bias_logvar = 'W'+str(i)+'_logvar', 'b'+str(i)+'_logvar'
        if i == 0:
            weights[weight_mean] = tf.Variable(xavier_initializer(n_in, architecture[i]), name=vname+weight_mean)
            weights[weight_logvar] = tf.Variable(tf.fill([n_in, architecture[i]], initVar), name=vname+weight_logvar)
        else:
            weights[weight_mean] = tf.Variable(xavier_initializer(architecture[i-1], architecture[i]), name=vname+weight_mean)
            weights[weight_logvar] = tf.Variable(tf.fill([architecture[i-1], architecture[i]], initVar), name=vname+weight_logvar)
        weights[bias_mean] = tf.Variable(tf.zeros(architecture[i]) + 1e-1, name=vname+bias_mean)
        weights[bias_logvar] = tf.Variable(tf.fill([architecture[i]], initVar), name=vname+bias_logvar)
    weights['Wout_mean'] = tf.Variable(xavier_initializer(architecture[-1], n_out), name=vname+'Wout_mean')
    weights['Wout_logvar'] = tf.Variable(tf.fill([architecture[-1], n_out], initVar), name=vname+'Wout_logvar')
    weights['bout_mean'] = tf.Variable(tf.zeros(n_out) + 1e-1, name=vname+'bout_mean')
    weights['bout_logvar'] = tf.Variable(tf.fill([n_out], value = initVar), name=vname+'bout_logvar')
    return weights


def xavier_initializer(fan_in, fan_out, constant=1): 
    """ Xavier initialization of network weights"""
    low = -constant*np.sqrt(6.0/(fan_in + fan_out)) 
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), 
        	              minval=low, maxval=high, 
            	              dtype=tf.float32)


def batch_norm_wrapper(inputs, weights, layer, training, decay = 0.999, epsilon=1e-3):
    layer = str(layer)

    if training is not None:
        batch_mean, batch_var = tf.nn.moments(inputs,[0])
        train_mean = tf.assign(weights['mean'+layer],
                               weights['mean'+layer] * decay + batch_mean * (1 - decay))
        train_var = tf.assign(weights['var'+layer],
                              weights['var'+layer] * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs,
                batch_mean, batch_var, weights['beta'+layer], weights['scale'+layer], epsilon)
    else:
        return tf.nn.batch_normalization(inputs,
            weights['mean'+layer], weights['var'+layer], weights['beta'+layer], weights['scale'+layer], epsilon)
