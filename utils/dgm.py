from __future__ import absolute_import
from __future__ import division 
from __future__ import print_function

import numpy as np

import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

import pdb

""" Module containing shared functions and structures for DGMS """

def _forward_pass_Gauss(x, weights, n_h, nonlinearity):
    """ Forward pass through the network with given weights - Gaussian output """
    for i, neurons in enumerate(n_h):
	weight_name, bias_name = 'W'+str(i), 'b'+str(i)
	if i==0:
	    h = nonlinearity(tf.add(tf.matmul(x, weights[weight_name]), weights[bias_name]))
	else:
	    h = nonlinearity(tf.add(tf.matmul(h, weights[weight_name]), weights[bias_name]))
    mean = tf.add(tf.matmul(h, weights['Wmean']), weights['bmean'])
    log_var = tf.add(tf.matmul(h, weights['Wvar']), weights['bvar'])
    return mean, log_var


def _forward_pass_Cat(x, weights, n_h, nonlinearity):
    """ Forward pass through network with given weights - Categorical output """
    return tf.nn.softmax(_forward_pass_Cat_logits(x, weights, n_h, nonlinearity))


def _forward_pass_Bernoulli(x, weights, n_h, nonlinearity):
    """ Forward pass through the network with given weights - Bernoulli output """
    return tf.nn.sigmoid(_forward_pass_Cat_logits(x, weights, n_h, nonlinearity))


def _forward_pass_Cat_logits(x, weights, n_h, nonlinearity):
    """ Forward pass through the network with weights as a dictionary """
    for i, neurons in enumerate(n_h):
    	weight_name, bias_name = 'W'+str(i), 'b'+str(i)
        if i==0:
	    h = nonlinearity(tf.add(tf.matmul(x, weights[weight_name]), weights[bias_name]))
        else:
            h = nonlinearity(tf.add(tf.matmul(h, weights[weight_name]), weights[bias_name]))
    logits = tf.add(tf.matmul(h, weights['Wout']), weights['bout'])
    return logits


def _gauss_kl(mean, sigma):
    """ compute the KL-divergence of a Gaussian against N(0,1) """
    mean_0, sigma_0 = tf.zeros_like(mean), tf.ones_like(sigma)
    mvnQ = tf.contrib.distributions.MultivariateNormalDiag(loc=mean, scale_diag=sigma)       ## tf 1.1.0
    prior = tf.contrib.distributions.MultivariateNormalDiag(loc=mean_0, scale_diag=sigma_0)  ## tf 1.1.0
    #mvnQ = tf.contrib.distributions.MultivariateNormalDiag(mean, tf.sqrt(sigma))              ## tf 0.11
    #prior = tf.contrib.distributions.MultivariateNormalDiag(mean_0, sigma_0)                  ## tf 0.11
    return tf.contrib.distributions.kl(mvnQ, prior)


def _init_Gauss_net(n_in, architecture, n_out):
    """ Initialize the weights of a 2-layer network parameterizeing a Gaussian """
    weights = {}
    for i, neurons in enumerate(architecture):
        weight_name, bias_name = 'W'+str(i), 'b'+str(i)
        if i == 0:
       	    weights[weight_name] = tf.Variable(xavier_initializer(n_in, architecture[i]))
    	else:
    	    weights[weight_name] = tf.Variable(xavier_initializer(architecture[i-1], architecture[i]))
    	weights[bias_name] = tf.Variable(tf.zeros(architecture[i]) + 1e-1)
    weights['Wmean'] = tf.Variable(xavier_initializer(architecture[-1], n_out))
    weights['bmean'] = tf.Variable(tf.zeros(n_out) + 1e-1)
    weights['Wvar'] = tf.Variable(xavier_initializer(architecture[-1], n_out))
    weights['bvar'] = tf.Variable(tf.zeros(n_out) + 1e-1)
    return weights


def _init_Cat_net(n_in, architecture, n_out):
    """ Initialize the weights of a 2-layer network parameterizeing a Categorical """
    weights = {}
    for i, neurons in enumerate(architecture):
        weight_name, bias_name = 'W'+str(i), 'b'+str(i)
        if i == 0:
            weights[weight_name] = tf.Variable(xavier_initializer(n_in, architecture[i]))
    	else:
    	    weights[weight_name] = tf.Variable(xavier_initializer(architecture[i-1], architecture[i]))
    	weights[bias_name] = tf.Variable(tf.zeros(architecture[i]) + 1e-1)
    weights['Wout'] = tf.Variable(xavier_initializer(architecture[-1], n_out))
    weights['bout'] = tf.Variable(tf.zeros(n_out) + 1e-1)
    return weights


def xavier_initializer(fan_in, fan_out, constant=1): 
    """ Xavier initialization of network weights"""
    low = -constant*np.sqrt(6.0/(fan_in + fan_out)) 
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), 
        	              minval=low, maxval=high, 
            	              dtype=tf.float32)
