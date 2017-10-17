from __future__ import absolute_import
from __future__ import division 
from __future__ import print_function

import numpy as np

import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.contrib.tensorboard.plugins import projector

import pdb

""" Module containing shared functions and structures for DGMS """

############# (log) Density functions ##############

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



############## Neural Network modules ##############

def initNetwork(n_in, n_hid, n_out, vname):
    weights = {}
    for layer, neurons in enumerate(n_hid):
        weight_name, bias_name = 'W'+str(layer), 'b'+str(layer)
        if layer == 0:
       	    weights[weight_name] = tf.get_variable(shape=[n_in, n_hid[layer]], name=vname+weight_name, initializer=xavier_initializer())
    	else:
    	    weights[weight_name] = tf.get_variable(shape=[n_hid[layer-1], n_hid[layer]], name=vname+weight_name, initializer=xavier_initializer(uniform=False))
    	weights[bias_name] = tf.Variable(tf.zeros(n_hid[layer]) + 1e-1, name=vname+bias_name)
    return weights

def initGaussNet(n_in, n_hid, n_out, vname):
    """ Initialize the weights of a network parameterizeing a Gaussian distribution"""
    weights = initNetwork(n_in, n_hid, n_out, vname)
    weights['Wmean'] = tf.get_variable(shape=[n_hid[-1], n_out], name=vname+'Wmean', initializer=xavier_initializer(uniform=False))
    weights['bmean'] = tf.Variable(tf.zeros(n_out) + 1e-1, name=vname+'bmean')
    weights['Wvar'] = tf.get_variable(shape=[n_hid[-1], n_out], name=vname+'Wvar', initializer=xavier_initializer(uniform=False))
    weights['bvar'] = tf.Variable(tf.zeros(n_out) + 1e-1, name=vname+'bvar')
    return weights

def initCatNet(n_in, n_hid, n_out, vname):
    """ Initialize the weights of a network parameterizeing a Gaussian distribution"""
    weights = initNetwork(n_in, n_hid, n_out, vname)
    weights['Wout'] = tf.get_variable(shape=[n_hid[-1], n_out], name=vname+'Wout', initializer=xavier_initializer(uniform=False))
    weights['bout'] = tf.Variable(tf.zeros([n_out]) + 1e-1, name=vname+'bout')
    return weights

def forwardPass(x, weights, n_h, nonlinearity, bn, training, scope, reuse):
    h = x
    for layer, neurons in enumerate(n_h):
	weight_name, bias_name = 'W'+str(layer), 'b'+str(layer)
	h = tf.matmul(h, weights[weight_name]) + weights[bias_name]
	if bn:
	    name = scope+'_bn'+str(layer)
	    h = tf.layers.batch_normalization(h, training=training, name=name, reuse=reuse)
	h = nonlinearity(h)
    return h	

def forwardPassGauss(x, weights, n_h, nonlinearity, bn, training=True, scope='scope', reuse=True):
    """ Forward pass through the network with given weights - Gaussian output """
    h = forwardPass(x, weights, n_h, nonlinearity, bn, training, scope, reuse)
    mean = tf.matmul(h, weights['Wmean']) + weights['bmean']
    log_var = tf.matmul(h, weights['Wvar']) + weights['bvar']
    return mean, log_var

def samplePassGauss(x, weights, n_h, nonlinearity, bn, mc_samps=1, training=True, scope='scope', reuse=True):
    """ Forward pass through the network with given weights - Gaussian sampling """
    mean, log_var = forwardPassGauss(x, weights, n_h, nonlinearity, bn, training, scope, reuse)
    shape = tf.concat([tf.constant([mc_samps,]), tf.shape(mean)], axis=-1)
    epsilon = tf.random_normal(shape, dtype=tf.float32)
    return mean, log_var, mean + tf.sqrt(tf.exp(log_var)) * epsilon

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


############## Bayesian Neural Network modules ############## 

def _forward_pass_Cat_logits_bnn(x, weights, q, n_h, nonlinearity, bn, training):
    """ Forward pass through a BNN and variational approximation with weights as dictionaries """
    h = x
    for layer, neurons in enumerate(n_h):
    	weight_name, bias_name = 'W'+str(layer), 'b'+str(layer)
    	weight_mean, bias_mean = 'W'+str(layer)+'_mean', 'b'+str(layer)+'_mean'
	htilde = tf.matmul(h, q[weight_mean]) + q[bias_mean]
	h = tf.matmul(h, weights[weight_name]) + weights[bias_name]
	if bn:
	    h = bayes_batch_norm(h, htilde, weights, q, layer, training)
        h = nonlinearity(h)
    logits = tf.matmul(h, weights['Wout']) + weights['bout']
    return logits

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

def _forward_pass_Cat_bnn(x, weights, q, n_h, nonlinearity, bn=False, training=True):
    """ Forward pass through network with given weights - Categorical output """
    return tf.nn.softmax(_forward_pass_Cat_logits_bnn(x, weights, q, n_h, nonlinearity, bn, training))

def bayes_batch_norm(inputs, var_inputs, weights, q, layer, training, decay=0.99, epsilon=1e-3):
    layer = str(layer)
    if training==True:
        batch_mean, batch_var = tf.nn.moments(var_inputs,[0])
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
