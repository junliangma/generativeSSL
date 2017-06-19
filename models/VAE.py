from __future__ import absolute_import
from __future__ import division 
from __future__ import print_function

from models.model import model

import numpy as np

import utils.dgm as dgm
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

import pdb, os

""" Standard VAE: P(Z)P(X|Z) """

class VAE(model):
    def __init__(self, Z_DIM=2, NUM_HIDDEN=[4,4], LEARNING_RATE=0.005, NONLINEARITY=tf.nn.relu, temperature_epochs=None, start_temp=None,
    		 BATCH_SIZE=16,NUM_EPOCHS=75, Z_SAMPLES=1, TYPE_PX='Gaussian', BINARIZE=False, LOGGING=False):

	super(VAE, self).__init__(Z_DIM, LEARNING_RATE, NUM_HIDDEN, TYPE_PX, NONLINEARITY, temperature_epochs, start_temp, NUM_EPOCHS, Z_SAMPLES, BINARIZE, LOGGING)

	self.BATCH_SIZE = BATCH_SIZE  # batch size
	self.name = 'vae'             # model name
	
                        

    def fit(self, Data):
    	self._process_data(Data)
	# Book keeping
    	self._create_placeholders()
	self._set_schedule()
    	self._initialize_networks()
    	
	# loss and statistics
	elbo, logpx, KLz = self._compute_ELBO(self.x_batch)
	weight_prior = self._weight_regularization()
	self.loss = -elbo + weight_prior/self.BATCH_SIZE
	test_elbo, _, _ = self._compute_ELBO(self.x_test)
	train_elbo, _, _ = self._compute_ELBO(self.x_train)
	# define optimizer
    	self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss, global_step=self.global_step)
	# summary statistics
	with tf.name_scope("summaries_elbo"):
	    tf.summary.scalar("ELBO", self.loss)
	    tf.summary.scalar("Train Loss", train_elbo)
	    tf.summary.scalar("Test Loss", test_elbo)
	    self.summary_op = tf.summary.merge_all()

    	# run and train
    	epoch, step, steps2epoch = 0, 0, np.round(self.TRAINING_SIZE / self.BATCH_SIZE)
    	with tf.Session() as sess:
    	    sess.run(tf.global_variables_initializer()) 
    	    total_loss = 0
	    saver = tf.train.Saver()
	    if self.LOGGING:
    	        writer = tf.summary.FileWriter(self.LOGDIR, sess.graph)
	    
    	    while epoch < self.NUM_EPOCHS:
    	    	x_batch, _ = Data.next_batch_regular(self.BATCH_SIZE)
		if self.BINARIZE:
		    x_batch = self._binarize(x)
	        feed_dict = {self.x_batch:x_batch, self.x_train:Data.data['x_train'], self.x_test:Data.data['x_test'], self.beta:self.schedule[epoch]}
    	    	_, loss_batch, px, kl, summary = sess.run([self.optimizer, self.loss, logpx, KLz, self.summary_op], feed_dict=feed_dict)
		

		if self.LOGGING:
		    writer.add_summary(summary, global_step=step)
    	    	total_loss += loss_batch
    	    	step = step + 1 

    	    	if step > steps2epoch:
    	    	    print('Epoch: {}, total: {:5.3f}, logpx: {:5.3f}, klz: {:5.3f}'.format(epoch,loss_batch, px, kl))
		    saver.save(sess, self.ckpt_dir, global_step=step)
		    trainELBO, testELBO = sess.run([train_elbo, test_elbo], feed_dict=feed_dict)
    	    	    #print('Epoch: {}, Train ELBO: {:5.3f}, Test ELBO: {:5.3f}'.format(epoch, trainELBO, testELBO))
		    #eb, lpx, kl = sess.run(self._compute_ELBO(self.x_batch), feed_dict)
		    #eb1 = lpx - self.schedule[epoch] * kl
    	    	    #print('Epoch: {}, Computed ELBO: {:5.3f}, Temperature ELBO: {:5.3f}, beta: {:5.3f}'.format(epoch, eb, eb1, self.schedule[epoch]))
    	    	    total_loss, step, epoch = 0.0, 0, epoch + 1
		   


	    
	    if self.LOGGING:
      	        writer.close()

    
    def encode(self, x):
    	mean, log_var = dgm._forward_pass_Gauss(x, self.Qx_z, self.NUM_HIDDEN, self.NONLINEARITY)
    	return mean, log_var


    def decode(self, z):
	if self.TYPE_PX=='Gaussian':
    	    mean, log_var = dgm._forward_pass_Gauss(z, self.Pz_x, self.NUM_HIDDEN, self.NONLINEARITY)
    	    return mean, log_var
	elif self.TYPE_PX=='Bernoulli':
	    pi = dgm._forward_pass_Bernoulli(z, self.Pz_x, self.NUM_HIDDEN, self.NONLINEARITY)
	    return pi

    def _sample_Z(self, x, n_samples=1):
    	""" Sample from Z with reparamterization """
	mean, log_var = dgm._forward_pass_Gauss(x, self.Qx_z, self.NUM_HIDDEN, self.NONLINEARITY)
	eps = tf.random_normal([tf.shape(x)[0], self.Z_DIM], dtype=tf.float32)
	return mean, log_var, mean + tf.sqrt(tf.exp(log_var)) * eps 

    def _compute_ELBO(self, x):
    	z_mean, z_log_var, z = self._sample_Z(x)
    	KLz = dgm._gauss_kl(z_mean, z_log_var)
	l_qz = dgm._gauss_logp(z, z_mean, z_log_var)
	l_pz = dgm._gauss_logp(z, tf.zeros_like(z), tf.ones_like(z)) 
    	l_px = self._compute_logpx(x, z)
	total_elbo = l_px - self.beta * (KLz) 
        return tf.reduce_mean(total_elbo), tf.reduce_mean(l_px), tf.reduce_mean(KLz)

    def encode_new(self, x):
        saver = tf.train.Saver()
	with tf.Session() as session:
	    ckpt = tf.train.get_checkpoint_state(self.ckpt_dir)
	    saver.restore(session, ckpt.model_checkpoint_path)
	    z_ = self.encode(x)
	    z = session.run(z_)
	return(z)

    def _generate_data(self, n_samps=int(1e3)):
	saver = tf.train.Saver()
  	with tf.Session() as session:
	    ckpt = tf.train.get_checkpoint_state(self.ckpt_dir)
	    saver.restore(session, ckpt.model_checkpoint_path)
	    z_ = np.random.normal(size=(n_samps, self.Z_DIM)).astype('float32')
	    if self.TYPE_PX=='Gaussian':
                xmean, xlogvar = self.decode(z_) 
		xs = tf.sqrt(tf.exp(xlogvar))
		eps = tf.random_normal([n_samps, self.X_DIM], dtype=tf.float32)
		x_ = xmean + xs * eps
            else:
                x_ = self.decode(z_) 
	    x_, mean, st = session.run([x_, xmean, xs])
	return x_, mean, st
    

    def _initialize_networks(self):
	if self.TYPE_PX=='Gaussian':
    	    self.Pz_x = dgm._init_Gauss_net(self.Z_DIM, self.NUM_HIDDEN, self.X_DIM, 'Pz_x_')
	elif self.TYPE_PX=='Bernoulli':
    	    self.Pz_x = dgm._init_Cat_net(self.Z_DIM, self.NUM_HIDDEN, self.X_DIM, 'Pz_x_')	   
    	self.Qx_z = dgm._init_Gauss_net(self.X_DIM, self.NUM_HIDDEN, self.Z_DIM, 'Qx_z_')

    
