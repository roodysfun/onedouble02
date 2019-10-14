from crypto_net import *

def trainData():
	p = dict()

	p['back_window_size'] = 128 # days to look back into the past
	p['front_window_size'] = 16 # days to predict from now into the future
	p['batch_size'] = 16

	if __name__ == '__main__':	

		db = CryptoData('/Users/kiriko/Documents/SIT/ICT1002/Project GUI/data/btc.csv')	

		inputs = tf.placeholder(tf.float32, [p['batch_size'], p['back_window_size'], db.C])

		p['learning_rate'] = tf.placeholder(tf.float32)
		p['labels'] = tf.placeholder(tf.float32, [None, p['front_window_size'], db.C])

		net = CryptoNet(inputs, p['front_window_size'])

		with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
			p['mid_loss'] = tf.reduce_mean(tf.abs(p['labels'][:,:,0] - net.outputs[:,:,0]))
			p['loss'] = tf.nn.l2_loss(p['labels'] - net.outputs)
			p['train'] = tf.train.AdamOptimizer(p['learning_rate'], epsilon=1e-8).minimize(p['loss'])	

		initial_learning_rate = 0.001
		min_learning_rate = 0.000001
		learning_rate_decay_limit = 0.0001

		num_batches_per_epoch = 300
		learning_decay = 30 * num_batches_per_epoch
		weights_decay_after = 5 * num_batches_per_epoch

		checkpoints_dir = 'checkpoints'
		if not os.path.exists(checkpoints_dir):
			os.makedirs(checkpoints_dir)

		def get_session():
			USE_GPU = 0
			if not USE_GPU:
				config = tf.ConfigProto(device_count = {'GPU': 0})
			else:
				config = tf.ConfigProto()
				config.gpu_options.allow_growth=True
			return tf.Session(config=config)

		with get_session() as session:
			
			checkpoint_num = 0
			
			num_batches = 4096
			batch_index = 0
			learning_step = 0
			session.run(tf.global_variables_initializer())

			while batch_index < num_batches:


				learning_rate = max(min_learning_rate, 
					initial_learning_rate * 0.5**(learning_step / learning_decay))
				learning_step += 1

				inputs, labels = db.train.get_batch(p['back_window_size'], p['front_window_size'], p['batch_size'])
				feed_dict = {
					net.inputs: inputs, 
					p['labels']: labels, 
					p['learning_rate']: learning_rate
				}

				batch_index += 1

				if batch_index and batch_index % 16 == 0:
					feed_dict[net.dropout] = False
					loss = session.run(p['loss'], feed_dict=feed_dict)
					mid_loss = session.run(p['mid_loss'], feed_dict=feed_dict)
					print('l2 loss: {}\nmid loss: {}'.format(loss, mid_loss))
				
				feed_dict[net.dropout] = True
				session.run(p['train'], feed_dict=feed_dict)

				if batch_index and batch_index % 512 == 0:
					if checkpoint_num == 0:
						with open(checkpoints_dir+'/train.txt', 'w') as f:
							f.write('')
					inputs, labels = db.validate.get_batch(p['back_window_size'], p['front_window_size'], p['batch_size'])
					feed_dict = {
						net.inputs: inputs, 
						p['labels']: labels, 
						p['learning_rate']: learning_rate, 
						net.dropout: False
					}
					loss = session.run(p['loss'], feed_dict=feed_dict)
					mid_loss = session.run(p['mid_loss'], feed_dict=feed_dict)
					print('----------------------------------------')
					print('l2 loss: {}\nmid loss: {}'.format(loss, mid_loss))
					print('saving checkpoint {}...'.format(checkpoint_num))
					net.npz_saver.save(session, checkpoints_dir+'/c-{}.npz'.format(checkpoint_num))
					with open(checkpoints_dir+'/train.txt', 'a') as f:
						f.write(' '.join(map(str, (checkpoint_num, loss)))+'\n')
					print('checkpoint saved!')
					print('----------------------------------------')
					checkpoint_num += 1

