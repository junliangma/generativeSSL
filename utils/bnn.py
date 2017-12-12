from __future__ import absolute_import
from __future__ import division 
from __future__ import print_function

import numpy as np
import utils.dgm as dgm

import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.contrib.distributions import RelaxedOneHotCategorical as Gumbel
import pdb

""" Module containing shared functions and structures for DGMS """

glorotNormal = xavier_initializer(uniform=False)
initNormal = tf.random_normal_initializer(stddev=1e-3)

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

def initCatBNN(n_in, n_hid, n_out, vname, initVar=-5.):
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
	wTilde[wName], wTilde[bName] = tf.squeeze(dgm.sampleNormal(meanW, logvarW,1),0), tf.squeeze(dgm.sampleNormal(meanB, logvarB,1),0)
    return wTilde

def sampleCatBNN(weights, n_hid):
    """ return a sample from weights of a categorical BNN """
    wTilde = sampleBNN(weights, n_hid)
    meanW, meanB = weights['Wout_mean'], weights['bout_mean']
    logvarW, logvarB = weights['Wout_logvar'], weights['bout_logvar']
    wTilde['Wout'], wTilde['bout'] = tf.squeeze(dgm.sampleNormal(meanW, logvarW,1),0), tf.squeeze(dgm.sampleNormal(meanB, logvarB,1),0)
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
	kl += tf.reduce_sum(dgm.standardNormalKL(wMean, wLv)) + tf.reduce_sum(dgm.standardNormalKL(bMean, bLv))
    return kl 
    
def klWCatBNN(q, W, n_hid, dist='Gaussian'):
    """ estimate KL(q||p) as logp(w) - logq(w) for a categorical BNN """
    l_pw, l_qw = klWBNN(q, W, n_hid, dist)
    w, b = W['Wout'], W['bout']
    wMean, bMean, wLv, bLv = q['Wout_mean'], q['bout_mean'], q['Wout_logvar'], q['bout_logvar']
    l_pw += tf.reduce_sum(dgm.standardNormalLogDensity(w)) + tf.reduce_sum(dgm.standardNormalLogDensity(b))
    l_qw += tf.reduce_sum(dgm.gaussianLogDensity(w,wMean,wLv)) + tf.reduce_sum(dgm.gaussianLogDensity(b,bMean,bLv))
    return l_pw - l_qw

def klWCatBNN_exact(q, n_hid):
    """ compute exact KL(q||p) with standard normal p(w) for a categorical BNN """
    kl = klBNN_exact(q, n_hid)
    wMean, bMean = q['Wout_mean'], tf.expand_dims(q['bout_mean'],1)
    wLv, bLv = q['Wout_logvar'], tf.expand_dims(q['bout_logvar'],1)
    kl += tf.reduce_sum(dgm.standardNormalKL(wMean, wLv)) + tf.reduce_sum(dgm.standardNormalKL(bMean, bLv))
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

def initTiedBNN(nn1, n_hid, initVar, vname, n_type):
    """ Return a network tied by differences to an existing network nn1 """
    weights = {}
    for layer, neurons in enumerate(n_hid):
        wName, bName = 'W'+str(layer), 'b'+str(layer)
        eps_weight, eps_bias = 'epsW'+str(layer), 'epsb'+str(layer)
        weights[eps_weight] = tf.get_variable(shape=nn1[wName].shape, name=vname+eps_weight, initializer=initNormal)
        weights[eps_bias] = tf.get_variable(shape=nn1[bName].shape, name=vname+eps_bias, initializer=initNormal)
        weights[wName+'_mean'] = tf.add(nn1[wName], weights[eps_weight], name=vname+wName+'_mean')
	weights[wName+'_logvar'] = tf.Variable(tf.fill(nn1[wName].shape, initVar), name=vname+wName+'_logvar')
        weights[bName+'_mean'] = tf.add(nn1[bName], weights[eps_bias], name=vname+bName+'_mean')
	weights[bName+'_logvar'] = tf.Variable(tf.fill(nn1[bName].shape, initVar), name=vname+bName+'_logvar')
    if n_type=='Gauss':
	""" TODO: Bayesian-Gaussian output layer """
	pass
    elif n_type=='Categorical':
        weights['eps_Wout'] = tf.get_variable(shape=nn1['Wout'].shape, name=vname+'eps_Wout', initializer=initNormal)
        weights['eps_bout'] = tf.get_variable(shape=nn1['bout'].shape, name=vname+'eps_bout', initializer=initNormal)
        weights['Wout_mean'] = tf.add(nn1['Wout'], weights['eps_Wout'], name=vname+'Wout')
        weights['bout_mean'] = tf.add(nn1['bout'], weights['eps_bout'], name=vname+'bout')
        weights['Wout_logvar'] = tf.Variable(tf.fill(nn1['Wout'].shape, initVar), name=vname+'Wout_logvar')
        weights['bout_logvar'] = tf.Variable(tf.fill(nn1['bout'].shape, initVar), name=vname+'bout_logvar')
    return weights

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

def initCatBNN(n_in, n_hid, n_out, vname, initVar=-5.):
    """ initialize a BNN with categorical output (classification """
    weights = initBNN(n_in, n_hid, n_out, initVar, vname)
    weights['Wout_mean'] = tf.get_variable(shape=[n_hid[-1], n_out], name=vname+'Wout_mean', initializer=xavier_initializer())
    weights['Wout_logvar'] = tf.Variable(tf.fill([n_hid[-1], n_out], initVar), name=vname+'Wout_logvar')
    weights['bout_mean'] = tf.Variable(tf.zeros([n_out]) + 1e-1, name=vname+'bout_mean')
    weights['bout_logvar'] = tf.Variable(tf.fill([n_out], value=initVar), name=vname+'bout_logvar')
    return weights
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
