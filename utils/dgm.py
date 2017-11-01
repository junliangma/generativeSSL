from __future__ import absolute_import
from __future__ import division 
from __future__ import print_function

import numpy as np

import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.contrib.tensorboard.plugins import projector

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

def sampleNormal(mu, logvar):
    """ return a reparameterized sample from a Gaussian distribution """
    eps = tf.random_normal(mu.get_shape(), dtype=tf.float32)
    return mu + eps * tf.sqrt(tf.exp(logvar))

def standardNormalKL(mu, logvar):
    """ compute the KL divergence between a Gaussian and standard normal """
    return -0.5 * tf.reduce_sum(1 + logvar - mu**2 - tf.exp(logvar), axis=-1)

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

def initBNN(n_in, n_hid, n_out, initVar, vname):
    """ initialize the weights that define a general Bayesian neural network """
    weights = {}
    for layer, neurons in enumerate(n_hid):
        weight_mean, bias_mean = 'W'+str(layer)+'_mean', 'b'+str(layer)+'_mean'
        weight_logvar, bias_logvar = 'W'+str(layer)+'_logvar', 'b'+str(layer)+'_logvar'
        if layer == 0:
            weights[weight_mean] = tf.get_variable(shape=[n_in, n_hid[layer]], name=vname+weight_mean, initializer=xavier_initializer())
            weights[weight_logvar] = tf.Variable(tf.fill([n_in,n_hid[layer]], initVar), name=vname+weight_logvar)
        else:
            weights[weight_mean] = tf.get_variable(shape=[n_hid[layer-1], n_hid[layer]],name=vname+weight_mean, initializer=xavier_initializer())
            weights[weight_logvar] = tf.Variable(tf.fill([n_hid[layer-1], n_hid[layer]], initVar), name=vname+weight_logvar)
        weights[bias_mean] = tf.Variable(tf.zeros([n_hid[layer]]) + 1e-1, name=vname+bias_mean)
        weights[bias_logvar] = tf.Variable(tf.fill([n_hid[layer]], initVar), name=vname+bias_logvar)
    return weights    

def initCatBNN(n_in, n_hid, n_out, vname, initVar=-5):
    """ initialize a BNN with categorical output (classification """
    weights = initBNN(n_in, n_hid, n_out, initVar, vname)
    weights['Wout_mean'] = tf.get_variable(shape=[n_hid[-1], n_out], name=vname+'Wout_mean', initializer=xavier_initializer())
    weights['Wout_logvar'] = tf.Variable(tf.fill([n_hid[-1], n_out], initVar), name=vname+'Wout_logvar')
    weights['bout_mean'] = tf.Variable(tf.zeros([n_out]) + 1e-1, name=vname+'bout_mean')
    weights['bout_logvar'] = tf.Variable(tf.fill([n_out], value=initVar), name=vname+'bout_logvar')
    return weights

def initGaussBNN(n_in, n_hid, n_out, vname, initVar=-5):
    """ TODO: initialize a BNN with Gaussian output (regression) """
    pass 

def sampleBNN(weights, n_hid):
    """ sample weights from a variational approximation """
    wTilde = {}
    for layer in range(len(n_hid)):
	wName, bName = 'W'+str(layer), 'b'+str(layer)
	meanW, meanB = weights['W'+str(layer)+'_mean'], weights['b'+str(layer)+'_mean']
	logvarW, logvarB = weights['W'+str(layer)+'_logvar'], weights['b'+str(layer)+'_logvar']
	wTilde[wName], wTilde[bName] = sampleNormal(meanW, logvarW), sampleNormal(meanB, logvarB)
    return wTilde

def sampleCatBNN(weights, n_hid):
    """ return a sample from weights of a categorical BNN """
    wTilde = sampleBNN(weights, n_hid)
    meanW, meanB = weights['Wout_mean'], weights['bout_mean']
    logvarW, logvarB = weights['Wout_logvar'], weights['bout_logvar']
    wTilde['Wout'], wTilde['bout'] = sampleNormal(meanW, logvarW), sampleNormal(meanB, logvarB)
    return wTilde

def sampleGaussBNN(weights, n_hid):
    """ return a sample from weights of a Gaussian BNN """
    pass

def klWBNN(q, W, n_hid, dist):
    """ estimate KL(q(w)||p(w)) as logp(w) - logq(w) 
	currently only p(w) = N(w;0,1) implemented """
    l_pw, l_qw = 0,0
    for layer, neurons in enumerate(n_hid):
        w, b =  W['W'+str(layer)], W['b'+str(layer)]
	wMean, bMean = q['W'+str(layer)+'_mean'], q['b'+str(layer)+'_mean']
        wLv, bLv = q['W'+str(layer)+'_logvar'], q['b'+str(layer)+'_logvar']
	l_pw += tf.reduce_sum(standardNormalLogDensity(w)) + tf.reduce_sum(standardNormalLogDensity(b))
	l_qw += tf.reduce_sum(gaussianLogDensity(w,wMean,wLv)) + tf.reduce_sum(gaussianLogDensity(b,bMean,bLv))
    return l_pw, l_qw

def klBNN_exact(q, n_hid):
    """ compute exact KL(q||N(0,1)) """
    kl = 0
    for layer, neurons in enumerate(n_hid):
	wMean, bMean = q['W'+str(layer)+'_mean'], tf.expand_dims(q['b'+str(layer)+'_mean'],1)
        wLv, bLv = q['W'+str(layer)+'_logvar'], tf.expand_dims(q['b'+str(layer)+'_logvar'],1)
	kl += tf.reduce_sum(standardNormalKL(wMean, wLv)) + tf.reduce_sum(standardNormalKL(bMean, bLv))
    return kl 
    
def klWCatBNN(q, W, n_hid, dist='Gaussian'):
    """ estimate KL(q||p) as logp(w) - logq(w) for a categorical BNN """
    l_pw, l_qw = klWBNN(q, W, n_hid, dist)
    w, b = W['Wout'], W['bout']
    wMean, bMean, wLv, bLv = q['Wout_mean'], q['bout_mean'], q['Wout_logvar'], q['bout_logvar']
    l_pw += tf.reduce_sum(standardNormalLogDensity(w)) + tf.reduce_sum(standardNormalLogDensity(b))
    l_qw += tf.reduce_sum(gaussianLogDensity(w,wMean,wLv)) + tf.reduce_sum(gaussianLogDensity(b,bMean,bLv))
    return l_pw - l_qw

def klWCatBNN_exact(q, n_hid):
    """ compute exact KL(q||p) with standard normal p(w) for a categorical BNN """
    kl = klBNN_exact(q, n_hid)
    wMean, bMean = q['Wout_mean'], tf.expand_dims(q['bout_mean'],1)
    wLv, bLv = q['Wout_logvar'], tf.expand_dims(q['bout_logvar'],1)
    kl += tf.reduce_sum(standardNormalKL(wMean, wLv)) + tf.reduce_sum(standardNormalKL(bMean, bLv))
    return kl

def averageVarBNN(q, n_hid):
    """ return the average (log) variance of variational distribution """
    totalVar, numParams = 0,0
    for layer in range(len(n_hid)):
        variances = tf.reshape(q['W'+str(layer)+'_logvar'], [-1])
        totalVar += tf.reduce_sum(tf.exp(variances))
        numParams += tf.cast(tf.shape(variances)[0], dtype=tf.float32)
        variances = tf.reshape(q['b'+str(layer)+'_logvar'], [-1])
        totalVar += tf.reduce_sum(tf.exp(variances))
        numParams += tf.cast(tf.shape(variances)[0], dtype=tf.float32)
    variances = tf.reshape(q['Wout_logvar'], [-1])
    totalVar += tf.reduce_sum(tf.exp(variances))
    numParams += tf.cast(tf.shape(variances)[0], tf.float32)
    variances = tf.reshape(q['bout_logvar'], [-1])
    totalVar += tf.reduce_sum(tf.exp(variances))
    numParams += tf.cast(tf.shape(variances)[0], dtype=tf.float32)    
    return totalVar/numParams 	
 
def bayesBatchNorm(inputs, var_inputs, weights, q, layer, training, decay=0.99, epsilon=1e-3):
    """ TODO: implement correctly - BatchNorm for BNNs """
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
