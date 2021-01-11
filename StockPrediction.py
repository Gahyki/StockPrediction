import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import os
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

tf.compat.v1.disable_eager_execution()
# Read csv file (df for dataframe)
df = pd.read_csv('KO.csv', usecols=['Date', 'Open', 'High', 'Low', 'Close'])

# # Displaying cvs data
# # Initialize size of graph
# plt.figure(figsize=(18, 9))
#
# # Average price of day
# plt.plot(range(df.shape[0]), (df['Low']+df['High'])/2.0)
#
# # X axis of every 14 days
# plt.xticks(range(0, df.shape[0], 14), df['Date'].loc[::14], rotation=45)
# plt.xlabel('Date', fontsize=18)
# plt.ylabel('Mid Price', fontsize=18)
# plt.show()

# Training set and test set
high = np.array(df.loc[:, 'High'])
low = np.array(df.loc[:, 'Low'])
mid = (high+low)/2
print(math.floor(len(df)*0.8))
train_data = mid[:math.floor(len(mid)*0.8)]
test_data = mid[math.floor(len(mid)*0.8):]

scaler = MinMaxScaler()
train_data = train_data.reshape(-1, 1)
test_data = test_data.reshape(-1, 1)

smoothing_window_size = 75
for di in range(0, len(train_data), smoothing_window_size):
    scaler.fit(train_data[di:di+smoothing_window_size, :])
    train_data[di:di+smoothing_window_size, :] = scaler.transform(train_data[di:di+smoothing_window_size, :])

for di in range(math.floor(len(mid)*0.8), len(train_data)):
    scaler.fit(train_data[di+smoothing_window_size:, :])
    train_data[di+smoothing_window_size:, :] = scaler.transform(train_data[di+smoothing_window_size:, :])

# Reshape both train and test data
train_data = train_data.reshape(-1)

# Normalize test data
test_data = scaler.transform(test_data).reshape(-1)

# Now perform exponential moving average smoothing
# So the data will have a smoother curve than the original ragged data
EMA = 0.0
gamma = 0.1
for ti in range(len(train_data)):
    EMA = gamma*train_data[ti] + (1-gamma)*EMA
    train_data[ti] = EMA

# Used for visualization and test purposes
all_mid_data = np.concatenate([train_data, test_data], axis=0)

window_size = 5
N = train_data.size
std_avg_predictions = []
std_avg_x = []
mse_errors = []

for pred_idx in range(window_size, N):
    if pred_idx >= N:
        date = dt.datetime.strptime('k', '%Y-%m-%d').date() + dt.timedelta(days=1)
    else:
        date = df.loc[pred_idx, 'Date']

    std_avg_predictions.append(np.mean(train_data[pred_idx-window_size:pred_idx]))
    mse_errors.append((std_avg_predictions[-1]-train_data[pred_idx])**2)
    std_avg_x.append(date)

print('MSE error for standard averaging: %.5f'%(0.5*np.mean(mse_errors)))

# # Display training set results
# plt.figure(figsize=(18, 9))
# plt.plot(range(df.shape[0]), all_mid_data, color='b', label='True')
# plt.plot(range(window_size, N), std_avg_predictions, color='orange', label='Prediction')
# plt.xticks(range(0, df.shape[0], 14), df['Date'].loc[::14], rotation=45)
# plt.xlabel('Date')
# plt.ylabel('Mid Price')
# plt.legend(fontsize=18)
# plt.show()


window_size = 5
N = train_data.size

run_avg_predictions = []
run_avg_x = []

mse_errors = []

running_mean = 0.0
run_avg_predictions.append(running_mean)

decay = 0.5

for pred_idx in range(1, N):

    running_mean = running_mean*decay + (1.0-decay)*train_data[pred_idx-1]
    run_avg_predictions.append(running_mean)
    mse_errors.append((run_avg_predictions[-1]-train_data[pred_idx])**2)
    run_avg_x.append(date)

print('MSE error for EMA averaging: %.5f'%(0.5*np.mean(mse_errors)))

# plt.figure(figsize=(18, 9))
# plt.plot(range(df.shape[0]), all_mid_data, color='b', label='True')
# plt.plot(range(0, N), run_avg_predictions, color='orange', label='Prediction')
# #plt.xticks(range(0,df.shape[0],50),df['Date'].loc[::50],rotation=45)
# plt.xlabel('Date')
# plt.ylabel('Mid Price')
# plt.legend(fontsize=18)
# plt.show()


class DataGeneratorSeq(object):

    def __init__(self, prices, batch_size, num_unroll):
        self._prices = prices
        self._prices_length = len(self._prices) - num_unroll
        self._batch_size = batch_size
        self._num_unroll = num_unroll
        self._segments = self._prices_length // self._batch_size
        self._cursor = [offset * self._segments for offset in range(self._batch_size)]

    def next_batch(self):

        batch_data = np.zeros(self._batch_size, dtype=np.float32)
        batch_labels = np.zeros(self._batch_size, dtype=np.float32)

        for b in range(self._batch_size):
            if self._cursor[b]+1>=self._prices_length:
                #self._cursor[b] = b * self._segments
                self._cursor[b] = np.random.randint(0, (b+1)*self._segments)

            batch_data[b] = self._prices[self._cursor[b]]
            batch_labels[b]= self._prices[self._cursor[b]+np.random.randint(0,5)]

            self._cursor[b] = (self._cursor[b]+1)%self._prices_length

        return batch_data, batch_labels

    def unroll_batches(self):

        unroll_data, unroll_labels = [], []
        init_data, init_label = None, None
        for ui in range(self._num_unroll):

            data, labels = self.next_batch()

            unroll_data.append(data)
            unroll_labels.append(labels)

        return unroll_data, unroll_labels

    def reset_indices(self):
        for b in range(self._batch_size):
            self._cursor[b] = np.random.randint(0, min((b+1)*self._segments,self._prices_length-1))



dg = DataGeneratorSeq(train_data,5,5)
u_data, u_labels = dg.unroll_batches()

for ui, (dat, lbl) in enumerate(zip(u_data, u_labels)):
    print('\n\nUnrolled index %d'%ui)
    dat_ind = dat
    lbl_ind = lbl
    print('\tInputs: ', dat)
    print('\n\tOutput:', lbl)

D = 1 # Dimensionality of the data. Since your data is 1-D this would be 1
num_unrollings = 5 # Number of time steps you look into the future.
batch_size = 25 # Number of samples in a batch
num_nodes = [200, 200, 150] # Number of hidden nodes in each layer of the deep LSTM stack we're using
n_layers = len(num_nodes) # number of layers
dropout = 0.2 # dropout amount

tf.compat.v1.reset_default_graph() # This is important in case you run this multiple times
train_inputs, train_outputs = [], []
for ui in range(num_unrollings):
    train_inputs.append(tf.compat.v1.placeholder(tf.float32, shape=(batch_size, D), name='train_inputs_%d'%ui))
    train_outputs.append(tf.compat.v1.placeholder(tf.float32, shape=(batch_size, 1), name='train_outputs_%d'%ui))

# for each layer, create each node
lstm_cells = [tf.keras.layers.LSTMCell(units=num_nodes[li]) for li in range(n_layers)]

drop_lstm_cells = [tf.compat.v1.nn.rnn_cell.DropoutWrapper(lstm,
        input_keep_prob=1.0,
        output_keep_prob=1.0-dropout,
        state_keep_prob=1.0-dropout)
    for lstm in lstm_cells]

drop_multi_cell = tf.keras.layers.StackedRNNCells(drop_lstm_cells)
multi_cell = tf.keras.layers.StackedRNNCells(lstm_cells)

w = tf.compat.v1.get_variable('w',shape=[num_nodes[-1], 1], initializer=tf.keras.initializers.GlorotUniform())
b = tf.compat.v1.get_variable('b',initializer=tf.random.uniform([1],-0.1,0.1))

# Create cell state and hidden state variables to maintain the state of the LSTM
c, h = [],[]
initial_state = []
for li in range(n_layers):
    c.append(tf.Variable(tf.zeros([batch_size, num_nodes[li]]), trainable=False))
    h.append(tf.Variable(tf.zeros([batch_size, num_nodes[li]]), trainable=False))
    initial_state.append(tf.compat.v1.nn.rnn_cell.LSTMStateTuple(c[li], h[li]))


# Do several tensor transofmations, because the function dynamic_rnn requires the output to be of
# a specific format. Read more at: https://www.tensorflow.org/api_docs/python/tf/nn/dynamic_rnn
all_inputs = tf.concat([tf.expand_dims(t, 0) for t in train_inputs], axis=0)

# all_outputs is [seq_length, batch_size, num_nodes]
all_lstm_outputs, state = tf.compat.v1.nn.dynamic_rnn(
    drop_multi_cell, all_inputs, initial_state=tuple(initial_state),
    time_major=True, dtype=tf.float32)
# state = tf.keras.layers.RNN(drop_multi_cell, time_major=True)(all_inputs, initial_state=tuple(initial_state))
# all_lstm_outputs = tf.keras.layers.RNN(drop_multi_cell, time_major=True)(all_inputs, initial_state=tuple(initial_state))[1:]

all_lstm_outputs = tf.reshape(all_lstm_outputs, [batch_size*num_unrollings, num_nodes[-1]])

all_outputs = tf.compat.v1.nn.xw_plus_b(all_lstm_outputs, w, b)

split_outputs = tf.split(all_outputs, num_unrollings, axis=0)