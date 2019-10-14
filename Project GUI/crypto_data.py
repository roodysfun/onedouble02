#import tensorflow as tf
import numpy as np
import random
import datetime 

class CryptoData(object):

	def __init__(self, filename='btc.csv'):
		
		import csv
		from collections import OrderedDict
		
		print('Setting up Crypto data...')

		month_names = 'jan,feb,mar,apr,may,jun,jul,aug,sep,oct,nov,dec'.split(',')
		def month_num(s):
			return month_names.index(s.lower()[:3]) + 1
		
		price_data = OrderedDict()
		price_data['mid'] = []
		price_data['high'] = []
		price_data['low'] = []
		price_data['vol'] = []
		price_data['cap'] = []

		day_data = OrderedDict()
		day_data['month'] = []
		day_data['weekday'] = []
		day_data['day'] = []

		self._last_date = None

		with open(filename) as f:
			rows = csv.reader(f)
			for ri, r in enumerate(rows):
				if ri > 0:
					try:
						d = r[0].split(' ')
						high = float(r[1])
						low = float(r[2])
						mid = (high + low) * 0.5
						vol = float(r[3].replace(',',''))
						cap = float(r[4].replace(',',''))
						d = datetime.datetime(int(d[2]), month_num(d[0]), int(d[1].strip(',')))
						if self._last_date is None:
							self._last_date = d
						month = float(d.month)
						weekday = float(d.weekday())
						day = float(d.day)
						price_data['mid'].append(mid)
						price_data['high'].append(high)
						price_data['low'].append(low)
						price_data['vol'].append(vol)
						price_data['cap'].append(cap)
						day_data['month'].append(month)
						day_data['weekday'].append(weekday)
						day_data['day'].append(day)
					except Exception, e:
						pass

		for k in price_data.keys():
			price_data[k] = price_data[k][::-1]
			price_data[k] = np.array(price_data[k], dtype=np.float32)
		
		for k in day_data.keys():
			day_data[k] = day_data[k][::-1]
			day_data[k] = np.array(day_data[k], dtype=np.float32)

		price_combined = np.array(price_data.values()).transpose()
		day_combined = np.array(day_data.values()).transpose()

		# discard the first day
		day_combined = day_combined[1:]

		# discard the first day
		returns_combined = (price_combined[1:] - price_combined[:-1]) / price_combined[:-1] 

		combined = np.concatenate([returns_combined, day_combined], axis=-1)

		self._combined_min = np.min(combined, axis=0)
		self._combined_max = np.max(combined, axis=0)
		self._combined_diff = self._combined_max - self._combined_min
		self._combined_normalized = (combined - self._combined_min) / self._combined_diff - 0.5

		for k in price_data.keys():
			price_data[k] = price_data[k][1:]
		
		for k in day_data.keys():
			day_data[k] = day_data[k][1:]
			
		self.price_data = price_data
		self.day_data = day_data

		self._mode = 'train'
		self._iters = {}
		self._data = {'train': [], 'validate': [], 'test': [], 'all': []}

		n = len(combined)
		for i in range(n):
			self._data['all'].append(i)
			if i < 0.7 * n:
				self._data['train'].append(i)
			elif i < 0.8 * n:
				self._data['validate'].append(i)
			else:
				self._data['test'].append(i)

		def get_random_iter(mode):
			while 1:
				order = np.arange(len(self._data[mode]))
				np.random.shuffle(order)
				for i in order:
					yield i
		
		for k in self._data:
			self._iters[k] = iter(get_random_iter(k))
			
		print('Crypto data setup complete!')

	@property
	def train(self):
		self._mode = 'train'
		return self

	@property
	def validate(self):
		self._mode = 'validate'
		return self
	
	@property
	def test(self):
		self._mode = 'test'
		return self

	@property
	def all(self):
		self._mode = 'all'
		return self

	@property
	def data(self):
		return self._data[self._mode]

	@property
	def C(self):
	    return self._combined_normalized.shape[-1]

	def __len__(self):
		return len(self._data[self._mode])

	def normalized_window(self):

		class LastWindow(object):
			
			def __init__(self, data, last_date, num_price_data_keys, date_norm_min, date_norm_diff):
				self._data = data[:]
				self._last_date = last_date
				self._D = num_price_data_keys
				self._date_norm_min = date_norm_min
				self._date_norm_diff = date_norm_diff

			def extend(self, outputs):
				shape = list(self._data.shape)
				n = shape[0]
				shape[0] = outputs.shape[0]
				self._data = np.concatenate([self._data, np.zeros(shape)], axis=0)
				self._data[n:, 0:outputs.shape[1]] = outputs
				i = n
				while i < self._data.shape[0]:
					self._last_date += datetime.timedelta(days=1)	
					d = self._last_date
					month = float(d.month)
					weekday = float(d.weekday())
					day = float(d.day)
					d_min = self._date_norm_min
					d_diff = self._date_norm_diff
					self._data[i, self._D + 0] = (month - d_min[0]) / d_diff[0] - 0.5
					self._data[i, self._D + 1] = (weekday - d_min[1]) / d_diff[1] - 0.5
					self._data[i, self._D + 2] = (day - d_min[2]) / d_diff[2] - 0.5
					i += 1
			
			@property
			def data(self):
				return self._data

		return LastWindow(
			self._combined_normalized,
			self._last_date,
			len(self.price_data),
			self._combined_min[len(self.price_data):],
			self._combined_diff[len(self.price_data):]
		)

	def unnormalized(self, outputs):
		return ((outputs + 0.5) * self._combined_diff) + self._combined_min

	def get_batch(self, back_window_size=56, front_window_size=1, batch_size=16):
		rn = random.randint
		T = self._combined_normalized
		N = batch_size
		C = T.shape[-1]
		W_back = back_window_size
		W_front = front_window_size
		X = np.zeros([N, W_back, C], dtype=np.float32)
		Y = np.zeros([N, W_front, C], dtype=np.float32)
		data = self._data[self._mode]
		next_int = self._iters[self._mode].next
		for bi in range(N):
			found = False
			while not found:
				i = data[next_int()]
				start = i - back_window_size
				end = i + front_window_size
				if start >= 0 and end <= T.shape[0]:
					found = True
					X[bi] = T[start:i]
					Y[bi] = T[i:end, 0:C]
		return X, Y


if __name__ == '__main__':
	db = CryptoData()
	X, Y = db.train.get_batch()
	#print X[0], Y[0]
	print db.normalized_window()