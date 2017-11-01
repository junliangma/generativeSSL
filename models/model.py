from __future__ import absolute_import 
from __future__ import division
from __future__ import print_function

import sys, os, pdb
import matplotlib.pyplot as plt
import numpy as np
import utils.dgm as dgm

import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

""" Super class for deep generative models """

class model(object):

    def __init__(self, n_x, n_y, n_z=2, n_hidden=[4], x_dist='Gaussian', nonlinearity=tf.nn.relu, batchnorm=False, mc_samples=1, alpha=0.1, l2_reg=0.3, ckpt=None):
	
	self.n_x, self.n_y = n_x, n_y    # data characterisits
	self.n_z = n_z                   # number of labeled dimensions
	self.n_hid = n_hidden            # network architectures
	self.nonlinearity = nonlinearity # activation function
	self.x_dist = x_dist             # likelihood for inputs
	self.bn = batchnorm              # use batch normalization
	self.mc_samples = mc_samples     # MC samples for estimation
	self.alpha = alpha               # additional penalty weight term
	self.l2_reg = l2_reg             # weight regularization scaling constant
	self.name = 'model'              # model name
	self.ckpt = ckpt 		 # preallocated checkpoint dir
	
	# placeholders for necessary values
	self.n, self.n_train = 1,1	 # initialize data size
  		
	self.build_model()
	self.loss = self.compute_loss()
	self.session = tf.Session()

    def train(self, Data, n_epochs, l_bs, u_bs, lr, eval_samps=None, temp_epochs=None, start_temp=0.0, binarize=False, logging=False, verbose=1):
	""" Method for training the models """
	self.data_init(Data, eval_samps, l_bs, u_bs)
	self.lr = self.set_learning_rate(lr)
	self.schedule = self.set_schedule(temp_epochs, start_temp, n_epochs)
	self.beta = tf.Variable(self.schedule[0], trainable=False, name='beta')
        ## define optimizer
	optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
	gvs = optimizer.compute_gradients(self.loss)
	capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	with tf.control_dependencies(update_ops):
	    self.optimizer = optimizer.apply_gradients(capped_gvs, global_step=self.global_step) 
	
	#update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	#with tf.control_dependencies(update_ops):
        #    self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss, global_step=self.global_step)
	
	self.compute_accuracies()
	self.train_acc = self.compute_acc(self.x_train, self.y_train)
	self.test_acc = self.compute_acc(self.x_test, self.y_test)

        ## initialize session and train
        max_acc, epoch, step = 0, 0, 0
        with self.session as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            if logging:
                writer = tf.summary.FileWriter(self.LOGDIR, sess.graph)

            while epoch < n_epochs:
                self.phase=True
                x_labeled, labels, x_unlabeled, _ = Data.next_batch(l_bs, u_bs)
                if binarize == True:
                    x_labeled, x_unlabeled = self.binarize(x_labeled), self.binarize(x_unlabeled)
                _, loss_batch = sess.run([self.optimizer, self.loss],
                                           feed_dict={self.x_l: x_labeled, self.y_l: labels,
                                           	self.x_u: x_unlabeled, self.x: x_labeled,
                                           	self.y: labels})
                if logging:
		    writer.add_summary(summary_elbo, global_step=self.global_step)

                if Data._epochs_unlabeled > epoch:
                    epoch += 1
                    fd = self._printing_feed_dict(Data, x_labeled, x_unlabeled, labels, eval_samps, binarize)
		    saver.save(sess, self.ckpt_dir, global_step=step+1)
		    if verbose == 1:
                        self.print_verbose1(epoch, fd, sess)
		    elif verbose == 2:
                        self.print_verbose2(epoch, fd, sess)

            if logging:
                writer.close()


    def encode_new(self, x):
        saver = tf.train.Saver()
        with self.session as session:
            ckpt = tf.train.get_checkpoint_state(self.ckpt_dir)
            saver.restore(session, ckpt.model_checkpoint_path)
            self.phase = False
            encoded = session.run([self.encoded], {self.x_new:x})
        return encoded[0]

    def predict_new(self, x):
        saver = tf.train.Saver()
        with tf.Session() as session:
            ckpt = tf.train.get_checkpoint_state(self.ckpt_dir)
            saver.restore(session, ckpt.model_checkpoint_path)
            self.phase = False
            preds = session.run(self.predictions, {self.x:x})
        return preds

### Every instance of model must implement these two methods ###

    def predict(self, x):
	pass

    def encode(self, x):
	pass

    def build_model(self):
	pass

    def compute_loss(self):
	pass

################################################################
    
    def weight_prior(self):
	weights = [V for V in tf.trainable_variables() if 'W' in V.name]
	return np.sum([tf.reduce_sum(dgm.standardNormalLogDensity(w)) for w in weights])

    def weight_regularization(self):
	weights = [V for V in tf.trainable_variables() if 'W' in V.name]
	return np.sum([tf.nn.l2_loss(w) for w in weights])	

    def data_init(self, Data, eval_samps, l_bs, u_bs):
	self._process_data(Data, eval_samps, l_bs, u_bs)

    def binarize(self, x):
	return np.random.binomial(1,x)

    def set_schedule(self, temp_epochs, start_temp, n_epochs):
	if not temp_epochs:
	    return np.ones((n_epochs, )).astype('float32')
	else:
	    warmup = np.expand_dims(np.linspace(start_temp, 1.0, temp_epochs),1)
	    plateau = np.ones((n_epochs-temp_epochs,1))
	    return np.ravel(np.vstack((warmup, plateau))).astype('float32')

    def set_learning_rate(self, lr):
	""" Set learning rate """
	self.global_step = tf.Variable(0, trainable=False, name='global_step')
	if len(lr) == 1:
	    return lr[0]
	else:
	    start_lr, rate, final_lr = lr
	    return tf.train.polynomial_decay(start_lr, self.global_step, rate, end_learning_rate=final_lr)     


    def _process_data(self, data, eval_samps, l_bs, u_bs):
        """ Extract relevant information from data_gen """
        self.n = data.N
        self.n_train = data.TRAIN_SIZE         # training set size
        self.n_test = data.TEST_SIZE           # test set size
        if eval_samps == None:
            self.eval_samps = self.n_train     # evaluation training set size
	    self.eval_samps_test = self.n_test # evaluation test set size
	else:
	    self.eval_samps_train = eval_samps
	    self.eval_samps_test = eval_samps
        self.n_l = data.NUM_LABELED            # number of labeled instances
        self.n_u = data.NUM_UNLABELED          # number of unlabeled instances
        self.data_name = data.NAME             # dataset being used   
        self._allocate_directory()             # logging directory
        self.alpha *= self.n_train/self.n_l    # weighting for additional term


    def create_placeholders(self):
        """ Create input/output placeholders """
        self.x_l = tf.placeholder(tf.float32, shape=[None, self.n_x], name='x_labeled')
        self.x_u = tf.placeholder(tf.float32, shape=[None, self.n_x], name='x_unlabeled')
        self.y_l = tf.placeholder(tf.float32, shape=[None, self.n_y], name='y_labeled')
        self.x_train = tf.placeholder(tf.float32, shape=[None, self.n_x], name='x_train')
        self.x_test = tf.placeholder(tf.float32, shape=[None, self.n_x], name='x_test')
        self.x = tf.placeholder(tf.float32, shape=[None, self.n_x], name='x')
        self.y_train = tf.placeholder(tf.float32, shape=[None, self.n_y], name='y_train')
        self.y_test = tf.placeholder(tf.float32, shape=[None, self.n_y], name='y_test')
        self.y = tf.placeholder(tf.float32, shape=[None, self.n_y], name='y')
	self.phase = True 

    def compute_accuracies(self):
        self.train_acc = self.compute_acc(self.x_train, self.y_train)
	self.test_acc = self.compute_acc(self.x_test, self.y_test)

    def _printing_feed_dict(self, Data, x_l, x_u, y, eval_samps, binarize):
	x_train, y_train = Data.sample_train(eval_samps)
	x_test, y_test = Data.sample_test(eval_samps)
	return {self.x_train:x_train, self.y_train:y_train,
                self.x_test:x_test, self.y_test:y_test,
                self.x_l:x_l, self.y_l:y, self.x_u: x_u}

    def _allocate_directory(self):
	if self.ckpt == None:
            self.LOGDIR = './graphs/'+self.name+'-'+self.data_name+'-'+str(self.n_z)+'-'+str(self.n_l)+'/'
            self.ckpt_dir = './ckpt/'+self.name+'-'+self.data_name+'-'+str(self.n_z)+'-'+str(self.n_l) + '/'
	else: 
            self.LOGDIR = 'graphs/'+self.ckpt+'/' 
	    self.ckpt_dir = './ckpt/' + self.ckpt + '/'
        if not os.path.isdir(self.ckpt_dir):
            os.mkdir(self.ckpt_dir)
        if not os.path.isdir(self.LOGDIR):
            os.mkdir(self.LOGDIR)



########## ACQUISTION FUNCTIONS ###########

    def _acquisition_new(self, x, acq_func):
	self.phase=False
        if acq_func == 'predictive_entropy':
            acquisition = self._predictive_entropy(x)
        elif acq_func == 'bald':
            acquisition = self._bald(x)
        elif acq_func == 'var_ratios':
            acquisition = self._variational_ratios(x)
	elif acq_func == 'random':
	    acquisition = self._random_query(x)
        saver = tf.train.Saver()
        with self.session as session:
            ckpt = tf.train.get_checkpoint_state(self.ckpt_dir)
            saver.restore(session, ckpt.model_checkpoint_path)
            acq = session.run(acquisition)
        return acq, np.argmax(acq)


    def _predictive_entropy(self, x):
        predictions = self.predict(x)
        return -tf.reduce_sum(predictions[0] * tf.log(1e-10+predictions[0]),axis=1)


    def _variational_ratios(self, x):
        predictions = self.predict(x)
        return 1 - tf.reduce_max(predictions[0], axis=1)

    def _bald(self, x):
	pred_samples = self.sample_y(x)
        predictions = tf.reduce_mean(pred_samples, axis=2)
        H = -tf.reduce_sum(predictions * tf.log(1e-10+predictions), axis=1)
        E = tf.reduce_mean(-tf.reduce_sum(pred_samples * tf.log(1e-10+pred_samples), axis=1), axis=1)
        return H - E

    def _random_query(self, x):
	return tf.random_normal(shape=[x.shape[0]])

###########################################




