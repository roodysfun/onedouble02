from crypto_data import *
import tensorflow as tf
import os

class NPZSaver(object):
	
	def __init__(self, net):
		self._net = net
	
	def save(self, session, f):
		np.savez_compressed(f, **dict((v.name, session.run(v)) for v in self._net.variables))
	
	def restore(self, session, f):
		kwds = np.load(f)
		for v in self._net.variables:
			if v.name in kwds:
				#print v.name
				session.run(v.assign(kwds[v.name]))

	def restore_pruned(self, session, f, threshold_percent=0.08):
		kwds = np.load(f)
		num_pruned = 0
		num_params = 0
		for v in self._net.variables:
			if v.name in kwds:
				t = kwds[v.name]
				num_params += len(t.ravel())
				if 'weights' in v.name or 'kernel' in v.name:
					q = abs(threshold_percent * float(np.std(t.ravel())))
					t = np.where(np.abs(t) < q, 0.0, t)
					num_pruned += np.sum(np.where(np.abs(t) < q, 1.0, 0.0))
				session.run(v.assign(t))
		print('Pruned: {}'.format(num_pruned / num_params))

class CryptoNet(object):

	def __init__(self, x, front_window_size):
		self.name = 'crypto_net'
		self.inputs = x
		self.dropout = tf.placeholder_with_default(False, shape=None)
		
		flatten = lambda x:tf.reshape(x, [-1, reduce(lambda a,b:a*b, x.shape.as_list()[1:])])
		


		with tf.variable_scope(self.name) as scope:
			C = x.shape[-1]
			
			x = tf.layers.conv1d(self.inputs, 16, 7, 2, name='c1_1', padding='VALID')
			x = tf.nn.tanh(x)
			x = tf.layers.conv1d(x, 16, 5, 2, name='c1_2', padding='VALID')
			x = tf.nn.tanh(x)
			x = tf.layers.conv1d(x, 16, 3, 1, name='c1_3', padding='VALID')
			x = tf.nn.tanh(x)

			x = flatten(x)
			x = tf.layers.dense(x, 8 * C, name='fc_1')
			x = tf.nn.tanh(x)
			x = tf.nn.dropout(x, tf.cond(self.dropout, lambda:0.25, lambda:1.0))

			x = tf.layers.dense(x, front_window_size * C, name='fc_last')
			x = tf.reshape(x, [-1, front_window_size, C])
			
			self.outputs = x
			
	@property
	def variables(self): return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name)
	@property
	def kernels(self): return [v for v in self.variables if 'kernel' in v.name[:v.name.rfind(':')].split('/')]
	@property
	def biases(self): return [v for v in self.variables if v.name[:v.name.rfind(':')].split('/')]
	@property
	def total_params(self): return sum(reduce(lambda a,b:a*b, v.get_shape().as_list(), 1) for v in self.variables)	
	@property
	def saver(self): return tf.train.Saver(self.variables)
	@property
	def npz_saver(self): return NPZSaver(self)