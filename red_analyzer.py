#python3 red_analyzer.py data/bitcoin_name.csv bitcoin_name

import pandas as pd
import numpy as np
import sys
import time
import seaborn as sns
import datetime
from sklearn.preprocessing import normalize

class Organizer(object):
	def __init__(self, csv_file, window_len = 10, split_percent = 0.1):
		self.csv_file = csv_file
		self.window_len = window_len
		self.split_percent = split_percent
	
	#Read data from csv , drop time coloumn
	def read(self):
		data = np.array(pd.read_csv(self.csv_file).drop('timestamp',1).values)
		#data = np.array(pd.read_csv(self.csv_file).values)[::-1]
		self.data = data
		self.size = len(data)

	#Get all btc value day by day until now(CoinmarketCap)
	def read_btc_day_by_day(self):
		bitcoin_market_info = pd.read_html("https://coinmarketcap.com/currencies/bitcoin/historical-data/?start=20130428&end="+time.strftime("%Y%m%d"))[0]
		bitcoin_market_info = bitcoin_market_info.assign(Date=pd.to_datetime(bitcoin_market_info['Date']))
		bitcoin_market_info.loc[bitcoin_market_info['Volume']=="-",'Volume']=0
		bitcoin_market_info['Volume'] = bitcoin_market_info['Volume'].astype('int64')
		self.bitcoin_market_info = bitcoin_market_info
		
	#Get all spesific coint value day by day until now(CoinmarketCap)
	def read_coin_type(self, coin_name):
		other_market_info = pd.read_html("https://coinmarketcap.com/currencies/" + coin_name + "/historical-data/?start=20130428&end="+time.strftime("%Y%m%d"))[0]
		other_market_info = other_market_info.assign(Date=pd.to_datetime(other_market_info['Date']))
		other_market_info.loc[other_market_info['Volume']=="-", 'Volume'] = 0
		other_market_info['Volume'] = other_market_info['Volume'].astype('int64')
		self.other_market_info = other_market_info
		self.coin_name = coin_name
		

	def create_merged_dataset(self):
		self.bitcoin_market_info.columns =[self.bitcoin_market_info.columns[0]]+['bt_'+i for i in self.bitcoin_market_info.columns[1:]]
		self.other_market_info.columns =[self.other_market_info.columns[0]]+[self.coin_name + "_"+i for i in self.other_market_info.columns[1:]]

		market_info = pd.merge(self.bitcoin_market_info,self.other_market_info, on=['Date'])
		market_info = market_info[market_info['Date']>='2016-01-01']
		#market_info = market_info[market_info['Date']>=self.other_market_info.values[-1][0]]
		
		
		for coins in ['bt_', self.coin_name + "_"]:
			kwargs = { coins+'day_diff': lambda x: (x[coins+'Close']-x[coins+'Open'])/x[coins+'Open']}
			market_info = market_info.assign(**kwargs)
		
		for coins in ['bt_', self.coin_name + "_"]:
			kwargs = { coins+'close_off_high': lambda x: 2*(x[coins+'High']- x[coins+'Close'])/(x[coins+'High']-x[coins+'Low'])-1,
												coins+'volatility': lambda x: (x[coins+'High']- x[coins+'Low'])/(x[coins+'Open'])}
			market_info = market_info.assign(**kwargs)

		model_data = market_info[['Date']+[coin+metric for coin in ['bt_', self.coin_name + "_"]
                                   for metric in ['Close','Volume','close_off_high','volatility']]]
		
		model_data = model_data.sort_values(by='Date')
	
		self.model_data = model_data

	#Split and Organize Gathered and Merged Data
	def split_merged_dataset(self, split_date='2017-06-01'):
		training_set, test_set = self.model_data[self.model_data['Date']<split_date], self.model_data[self.model_data['Date']>=split_date]
		training_set = training_set.drop('Date', 1)
		test_set = test_set.drop('Date', 1)
		
		LSTM_training_inputs = []
		norm_cols = [coin+metric for coin in ['bt_', self.coin_name + "_"] for metric in ['Close','Volume']]
		for i in range(len(training_set)-self.window_len):
			temp_set = training_set[i:(i+self.window_len)].copy()
			for col in norm_cols:
				temp_set.loc[:, col] = temp_set[col]/temp_set[col].iloc[0] - 1
			LSTM_training_inputs.append(temp_set)
		LSTM_training_outputs = (training_set[self.coin_name + '_Close'][self.window_len:].values/training_set[self.coin_name + '_Close'][:-self.window_len].values) - 1
		
		LSTM_test_inputs = []
		for i in range(len(test_set)-self.window_len):
			temp_set = test_set[i:(i+self.window_len)].copy()
			for col in norm_cols:
				temp_set.loc[:, col] = temp_set[col]/temp_set[col].iloc[0] - 1
			LSTM_test_inputs.append(temp_set)
		LSTM_test_outputs = (test_set[self.coin_name + '_Close'][self.window_len:].values/test_set[self.coin_name + '_Close'][:-self.window_len].values)-1
		LSTM_training_inputs = [np.array(LSTM_training_input) for LSTM_training_input in LSTM_training_inputs]
		LSTM_training_inputs = np.array(LSTM_training_inputs)

		LSTM_test_inputs = [np.array(LSTM_test_inputs) for LSTM_test_inputs in LSTM_test_inputs]
		LSTM_test_inputs = np.array(LSTM_test_inputs)

		self.train_x = LSTM_training_inputs
		self.train_y = LSTM_training_outputs
		self.test_x = LSTM_test_inputs
		self.test_y = LSTM_test_outputs

	#Split Data to test and train sets(BITTREX DATA FROM CSV)
	def split(self):
		test_size = int(self.size * self.split_percent)
		self.train_data = self.data[:self.size - test_size]
		self.test_data = self.data[-test_size:]
		
	#Dataset organization for lstm(BITTREX DATA FROM CSV)
	def organize(self):
		inputs_and_outputs = []
		for d_set in [self.train_data, self.test_data]:
			inputs = []
			for i in range(len(d_set) - self.window_len):
				temp_set = np.array(d_set[i:(i + self.window_len)].copy())
				norm = temp_set[:,[3,4,5,6,7,8,9,12]]
				norm = norm / (norm.max(axis=0) + 0.0000001)
				#pred = temp_set[:, [10]]
				#pred /= 100.0
				#temp_set = np.concatenate((temp_set[:, [0,1,2]], norm[:, [0,1,2,3,4,5,6]], pred, temp_set[:,[11]], norm[:,[7]]), axis=1)
				temp_set = np.concatenate((temp_set[:, [0,1,2]], norm[:, [0,1,2,3,4,5,6]], temp_set[:,[10,11]], norm[:,[7]]), axis=1)
				inputs.append(temp_set)
			outputs = [elem[-3] for elem in d_set[self.window_len:]]
			inputs_and_outputs.append((inputs, outputs))
		inputs_and_outputs = np.array(inputs_and_outputs)

		self.train_x = np.array(inputs_and_outputs[0][0])
		self.train_y = np.array(inputs_and_outputs[0][1])
		self.test_x = np.array(inputs_and_outputs[1][0])
		self.test_y = np.array(inputs_and_outputs[1][1])

	def print_first_5(self):
		print("Size : {}".format(len(self.data)))
		print(self.data[:5])

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.models import load_model

class DeepRecurrentPredictor(object):
	def __init__(self, rand_seed=755):
		self.rand_seed = rand_seed
		
	def create(self, inputs, output_size, neurons, act_func = "linear", dropout=0.25, loss="mse", optimizer="adam"):
		model = Sequential()
		model.add(LSTM(neurons, input_shape=(inputs.shape[1], inputs.shape[2])))
		model.add(Dropout(dropout))
		model.add(Dense(units=output_size))
		model.add(Activation(act_func))
		model.compile(loss=loss, optimizer=optimizer)
		self.model = model
		return model
		
	def train(self, train_inputs, train_outputs, epoch=50, model_name="btc"):
		np.random.seed(self.rand_seed)
		self.model.fit(train_inputs, train_outputs, epochs=epoch, batch_size=10, verbose=1, shuffle=True)
		self.model.save(model_name + ".h5")
	
	def test_after_load(self, model_name, test_inputs, test_outputs):
		temp_model = load_model(model_name + ".h5")
		self.model = temp_model
		results = temp_model.predict(test_inputs)
		accuracy = np.mean(abs(np.transpose(results) - test_outputs))
		print("accuracy : {}".format(accuracy))
		return accuracy

	def test(self, test_inputs, test_outputs):
		results = self.model.predict(test_inputs)
		accuracy = np.mean(abs(np.transpose(results)- test_outputs))
		print("accuracy : {}".format(accuracy))
		return accuracy

	def test_merged(self, test_inputs, test_outputs, model_name="ripple"):
		temp_model = load_model(model_name + ".h5")
		results = self.model.predict(test_inputs)
		preds = np.mean(abs(np.transpose(results)- test_outputs))
		print("preds : {}".format(preds))

	def predict(self, test_input):
		result = self.model.predict(test_input, batch_size=1)
		#print("prediction value : {}".format(result))
		return result
		
if __name__ == "__main__":
	organizer = Organizer(sys.argv[1], window_len=15)
	model_save_name = sys.argv[2]
	#BITTREX CSV
	organizer.read()
	#organizer.print_first_5()
	organizer.split()
	organizer.organize()
	'''
	#COIN MARKET CAP
	organizer.read_btc_day_by_day()
	organizer.read_coin_type('ripple')
	organizer.create_merged_dataset()
	organizer.split_merged_dataset()
	'''	
	deep_predictor = DeepRecurrentPredictor()
	deep_predictor.create(organizer.train_x, 1, 128, optimizer="adam")
	print("training started")
	deep_predictor.train(organizer.train_x, organizer.train_y, model_name=model_save_name, epoch=10)
	deep_predictor.test_merged(organizer.test_x, organizer.test_y, model_name=model_save_name)

	#BITTREX VALUES
	deep_predictor.test_after_load( model_save_name, organizer.test_x, organizer.test_y)
	starting_money = 1000	
	prev = organizer.train_y[-1]
	pred_prev = organizer.train_y[-1]
	same_counter = 0
	total=0
	for index, (x,y) in enumerate(zip(organizer.test_x, organizer.test_y)):
		res = deep_predictor.predict(np.array([x]))
		pred_is_inc = (res[0][0] - pred_prev) > 0
		curr_is_inc = (y - prev) > 0
		#print("prediction increasing : {} , real increasing : {}".format(pred_is_inc, curr_is_inc))
		#print("pred : {} - actual : {}".format(res[0][0], y))
		if(pred_is_inc == curr_is_inc):
			same_counter += 1
		prev = y
		pred_prev = res[0][0]
		total += 1
		#input()
	print("correct/all : {}/{}".format(same_counter, total))
