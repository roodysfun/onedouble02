import math, datetime
import tensorflow as tf
from crypto_net import *

db = CryptoData('/Users/kiriko/Documents/SIT/ICT1002/crypto_pred/btc.csv')

p = dict()

p['back_window_size'] = 128 # days to look back into the past
p['front_window_size'] = 16 # days to predict from now into the future

inputs = tf.placeholder(tf.float32, [1, p['back_window_size'], db.C])

net = CryptoNet(inputs, p['front_window_size'])

p['labels'] = tf.placeholder(tf.float32, [None, p['front_window_size'], db.C])

# directory
checkpoints_dir = 'checkpoints'
checkpoint_num = min([l.split(' ') for l in 
	open(checkpoints_dir+'/train.txt').read().split('\n') if l], 
	key=lambda l: float(l[1]))[0]

with tf.Session() as session:
	session.run(tf.global_variables_initializer())
	net.npz_saver.restore(session, checkpoints_dir+'/c-{}.npz'.format(checkpoint_num))

	mids = list(db.price_data['mid'])

	window = db.normalized_window()
	
	predicted_mids = []
	
	last_mid = db.price_data['mid'][-1]
	
	# we can use the predicted outputs to predict the next outputs... but the accuracy will drop
	for t in range(5): 

		feed_dict = {
			net.inputs: [window.data[-p['back_window_size']:]], 
			net.dropout: True
		}
		outputs = session.run(net.outputs, feed_dict=feed_dict)[0]
		unormalized_outputs = db.unnormalized(outputs)
		
		for o in unormalized_outputs:
			mid = o[0] * last_mid + last_mid
			predicted_mids.append(mid)
			last_mid = mid
		window.extend(outputs)


	import matplotlib.pyplot as plt

	n = len(db.price_data['mid'])
	plt.plot(np.arange(n), db.price_data['mid'])
	plt.plot(np.arange(n, n + len(predicted_mids)), predicted_mids)
	plt.show()




	#net.