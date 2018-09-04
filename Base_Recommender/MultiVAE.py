import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import apply_regularization, l2_regularizer

import bottleneck as bn

slim=tf.contrib.slim
Bernoulli = tf.contrib.distributions.Bernoulli

class GumbelSoftmax(object):
	def __init__(self, p_dims, q_dims=None, lam=0.01, lr=1e-3, random_seed=None, N = 30, K = 10):

		print 'Using N = ', N, 'K = ', K, '\n'

		self.p_dims = p_dims
		if q_dims is None:
			self.q_dims = p_dims[::-1]
		else:
			assert q_dims[0] == p_dims[-1], "Input and output dimension must equal each other for autoencoders."
			assert q_dims[-1] == p_dims[0], "Latent dimension for p- and q-network mismatches."
			self.q_dims = q_dims
		self.dims = self.q_dims + self.p_dims[1:]
		
		self.lam = lam
		self.lr = lr
		self.random_seed = random_seed

		self.N = N
		self.K = K

		self.is_training_ph = tf.placeholder_with_default(0., shape=None)
		self.anneal_ph = tf.placeholder_with_default(1., shape=None)

		self.construct_placeholders()

	def construct_placeholders(self):
		# print('dim:', self.dims[0])
		self.input_ph = tf.placeholder(dtype=tf.float32, shape=[None, self.dims[0]])
		self.keep_prob_ph = tf.placeholder_with_default(0.75, shape=None)

	def build_graph(self):

		# variational posterior q(y|x), i.e. the encoder (shape=(batch_size,200))
		net = slim.stack(self.input_ph, slim.fully_connected, [512,256])

		# unnormalized logits for N separate K-categorical distributions (shape=(batch_size*N,K))
		logits_y = tf.reshape(slim.fully_connected(net, self.K * self.N, activation_fn=None),[-1, self.K])

		q_y = tf.nn.softmax(logits_y)
		log_q_y = tf.log(q_y + 1e-20)

		# temperature
		self.tau = tf.Variable(5.0, name="temperature")


		# sample and reshape back (shape=(batch_size,N,K))
		# set hard=True for ST Gumbel-Softmax
		y = tf.reshape(self.gumbel_softmax(logits_y, self.tau, hard=False),[-1, self.N, self.K])

		# generative model p(x|y), i.e. the decoder (shape=(batch_size,200))
		net = slim.stack(slim.flatten(y), slim.fully_connected, [256,512])

		logits_x = slim.fully_connected(net, self.dims[0], activation_fn=None)

		p_x = Bernoulli(logits = logits_x)

		# loss and train ops
		kl_tmp = tf.reshape(q_y*(log_q_y-tf.log(1.0/self.K)),[-1, self.N, self.K])
		KL = tf.reduce_sum(kl_tmp,[1,2])
		elbo = tf.reduce_sum(p_x.log_prob(self.input_ph),1) - KL

		loss = tf.reduce_mean(-elbo)
		lr = tf.constant(0.001)
		train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss, var_list=slim.get_model_variables())

		# add summary statistics
		tf.summary.scalar('negative_multi_ll', -tf.reduce_sum(p_x.log_prob(self.input_ph),1))
		tf.summary.scalar('KL', KL)
		tf.summary.scalar('neg_ELBO', -elbo)
		merged = tf.summary.merge_all()

		return tf.train.Saver(), logits_x, -elbo, train_op, merged



	def sample_gumbel(self, shape, eps=1e-20): 
		"""Sample from Gumbel(0, 1)"""
		U = tf.random_uniform(shape,minval=0,maxval=1)
		return -tf.log(-tf.log(U + eps) + eps)

	def gumbel_softmax_sample(self, logits, temperature): 
		""" Draw a sample from the Gumbel-Softmax distribution"""
		y = logits + self.sample_gumbel(tf.shape(logits))
		return tf.nn.softmax( y / temperature)

	def gumbel_softmax(self, logits, temperature, hard=False):
		"""Sample from the Gumbel-Softmax distribution and optionally discretize.
		Args:
			logits: [batch_size, n_class] unnormalized log-probs
			temperature: non-negative scalar
			hard: if True, take argmax, but differentiate w.r.t. soft sample y
		Returns:
			[batch_size, n_class] sample from the Gumbel-Softmax distribution.
			If hard=True, then the returned sample will be one-hot, otherwise it will
			be a probabilitiy distribution that sums to 1 across classes
		"""
		y = self.gumbel_softmax_sample(logits, temperature)
		if hard:
			k = tf.shape(logits)[-1]
			#y_hard = tf.cast(tf.one_hot(tf.argmax(y,1),k), y.dtype)
			y_hard = tf.cast(tf.equal(y,tf.reduce_max(y,1,keep_dims=True)),y.dtype)
			y = tf.stop_gradient(y_hard - y) + y
		return y



class MultiDAE(object):
	def __init__(self, p_dims, q_dims=None, lam=0.01, lr=1e-3, random_seed=None):
		self.p_dims = p_dims
		if q_dims is None:
			self.q_dims = p_dims[::-1]
		else:
			assert q_dims[0] == p_dims[-1], "Input and output dimension must equal each other for autoencoders."
			assert q_dims[-1] == p_dims[0], "Latent dimension for p- and q-network mismatches."
			self.q_dims = q_dims
		self.dims = self.q_dims + self.p_dims[1:]
		
		self.lam = lam
		self.lr = lr
		self.random_seed = random_seed

		self.construct_placeholders()

	def construct_placeholders(self):        
		self.input_ph = tf.placeholder(
			dtype=tf.float32, shape=[None, self.dims[0]])
		self.keep_prob_ph = tf.placeholder_with_default(0.75, shape=None)

	def build_graph(self):

		self.construct_weights()

		saver, logits = self.forward_pass()
		log_softmax_var = tf.nn.log_softmax(logits)

		# per-user average negative log-likelihood
		neg_ll = -tf.reduce_mean(tf.reduce_sum(
			log_softmax_var * self.input_ph, axis=1))
		# apply regularization to weights
		reg = l2_regularizer(self.lam)
		reg_var = apply_regularization(reg, self.weights)
		# tensorflow l2 regularization multiply 0.5 to the l2 norm
		# multiply 2 so that it is back in the same scale
		loss = neg_ll + 2 * reg_var
		
		train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)

		# add summary statistics
		tf.summary.scalar('negative_multi_ll', neg_ll)
		tf.summary.scalar('loss', loss)
		merged = tf.summary.merge_all()
		return saver, logits, loss, train_op, merged

	def forward_pass(self):
		# construct forward graph        
		h = tf.nn.l2_normalize(self.input_ph, 1)
		h = tf.nn.dropout(h, self.keep_prob_ph)
		
		for i, (w, b) in enumerate(zip(self.weights, self.biases)):
			h = tf.matmul(h, w) + b
			
			if i != len(self.weights) - 1:
				h = tf.nn.tanh(h)
		return tf.train.Saver(), h

	def construct_weights(self):

		self.weights = []
		self.biases = []
		
		# define weights
		for i, (d_in, d_out) in enumerate(zip(self.dims[:-1], self.dims[1:])):
			weight_key = "weight_{}to{}".format(i, i+1)
			bias_key = "bias_{}".format(i+1)
			
			self.weights.append(tf.get_variable(
				name=weight_key, shape=[d_in, d_out],
				initializer=tf.contrib.layers.xavier_initializer(
					seed=self.random_seed)))
			
			self.biases.append(tf.get_variable(
				name=bias_key, shape=[d_out],
				initializer=tf.truncated_normal_initializer(
					stddev=0.001, seed=self.random_seed)))
			
			# add summary stats
			tf.summary.histogram(weight_key, self.weights[-1])
			tf.summary.histogram(bias_key, self.biases[-1])


class MultiVAE(MultiDAE):

	def construct_placeholders(self):
		super(MultiVAE, self).construct_placeholders()

		# placeholders with default values when scoring
		self.is_training_ph = tf.placeholder_with_default(0., shape=None)
		self.anneal_ph = tf.placeholder_with_default(1., shape=None)
		
	def build_graph(self):
		self._construct_weights()

		saver, logits, KL = self.forward_pass()
		log_softmax_var = tf.nn.log_softmax(logits)

		neg_ll = -tf.reduce_mean(tf.reduce_sum(
			log_softmax_var * self.input_ph,
			axis=-1))
		# apply regularization to weights
		reg = l2_regularizer(self.lam)
		
		reg_var = apply_regularization(reg, self.weights_q + self.weights_p)
		# tensorflow l2 regularization multiply 0.5 to the l2 norm
		# multiply 2 so that it is back in the same scale
		neg_ELBO = neg_ll + self.anneal_ph * KL + 2 * reg_var
		
		train_op = tf.train.AdamOptimizer(self.lr).minimize(neg_ELBO)

		# add summary statistics
		tf.summary.scalar('negative_multi_ll', neg_ll)
		tf.summary.scalar('KL', KL)
		tf.summary.scalar('neg_ELBO_train', neg_ELBO)
		merged = tf.summary.merge_all()

		params = []

		for elem in self.weights_q:
			params.append(elem)

		for elem in self.weights_p:
			params.append(elem)

		for elem in self.biases_q:
			params.append(elem)

		for elem in self.biases_p:
			params.append(elem)

		return saver, tf.nn.softmax(logits), neg_ELBO, train_op, merged, params
	
	def q_graph(self):
		mu_q, std_q, KL = None, None, None
		
		h = tf.nn.l2_normalize(self.input_ph, 1)
		h = tf.nn.dropout(h, self.keep_prob_ph)
		
		for i, (w, b) in enumerate(zip(self.weights_q, self.biases_q)):
			h = tf.matmul(h, w) + b
			
			if i != len(self.weights_q) - 1:
				h = tf.nn.tanh(h)
			else:
				mu_q = h[:, :self.q_dims[-1]]
				logvar_q = h[:, self.q_dims[-1]:]

				std_q = tf.exp(0.5 * logvar_q)
				KL = tf.reduce_mean(tf.reduce_sum(
						0.5 * (-logvar_q + tf.exp(logvar_q) + mu_q**2 - 1), axis=1))
		return mu_q, std_q, KL

	def p_graph(self, z):
		h = z
		
		for i, (w, b) in enumerate(zip(self.weights_p, self.biases_p)):
			h = tf.matmul(h, w) + b
			
			if i != len(self.weights_p) - 1:
				h = tf.nn.tanh(h)
		return h

	def forward_pass(self):
		# q-network
		mu_q, std_q, KL = self.q_graph()
		epsilon = tf.random_normal(tf.shape(std_q))

		sampled_z = mu_q + self.is_training_ph *\
			epsilon * std_q

		# p-network
		logits = self.p_graph(sampled_z)
		
		return tf.train.Saver(), logits, KL

	def _construct_weights(self):
		self.weights_q, self.biases_q = [], []
		
		for i, (d_in, d_out) in enumerate(zip(self.q_dims[:-1], self.q_dims[1:])):
			if i == len(self.q_dims[:-1]) - 1:
				# we need two sets of parameters for mean and variance,
				# respectively
				d_out *= 2
			weight_key = "weight_q_{}to{}".format(i, i+1)
			bias_key = "bias_q_{}".format(i+1)
			
			self.weights_q.append(tf.get_variable(
				name=weight_key, shape=[d_in, d_out],
				initializer=tf.contrib.layers.xavier_initializer(
					seed=self.random_seed)))
			
			self.biases_q.append(tf.get_variable(
				name=bias_key, shape=[d_out],
				initializer=tf.truncated_normal_initializer(
					stddev=0.001, seed=self.random_seed)))
			
			# add summary stats
			tf.summary.histogram(weight_key, self.weights_q[-1])
			tf.summary.histogram(bias_key, self.biases_q[-1])
			
		self.weights_p, self.biases_p = [], []

		for i, (d_in, d_out) in enumerate(zip(self.p_dims[:-1], self.p_dims[1:])):
			weight_key = "weight_p_{}to{}".format(i, i+1)
			bias_key = "bias_p_{}".format(i+1)
			self.weights_p.append(tf.get_variable(
				name=weight_key, shape=[d_in, d_out],
				initializer=tf.contrib.layers.xavier_initializer(
					seed=self.random_seed)))
			
			self.biases_p.append(tf.get_variable(
				name=bias_key, shape=[d_out],
				initializer=tf.truncated_normal_initializer(
					stddev=0.001, seed=self.random_seed)))
			
			# add summary stats
			tf.summary.histogram(weight_key, self.weights_p[-1])
			tf.summary.histogram(bias_key, self.biases_p[-1])

