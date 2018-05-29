import numpy as np
import pandas as pd
import math
import sklearn
from sklearn.preprocessing import MinMaxScaler
import datetime
import os
import matplotlib.pyplot as plt
import tensorflow as tf



valid_set_size_percentage = 5
test_set_size_percentage = 5
seq_len = 30

df = pd.read_csv('data/prices2.csv')
df = df.fillna(method='pad')
df.set_index('Date',inplace=True)

valid_set_size = int(np.round(valid_set_size_percentage/100*df.shape[0]));  
test_set_size = int(np.round(test_set_size_percentage/100*df.shape[0]));
train_set_size = df.shape[0] - (valid_set_size + test_set_size);


train_df = df[:train_set_size]
valid_df = df[train_set_size:train_set_size+valid_set_size]
test_df = df[train_set_size+valid_set_size:]


scaler = MinMaxScaler()

train_df.loc[:,'Open'] = scaler.fit_transform(train_df.Open.values.reshape(-1,1))
train_df.loc[:,'High'] = scaler.fit_transform(train_df.High.values.reshape(-1,1))
train_df.loc[:,'Low'] = scaler.fit_transform(train_df.Low.values.reshape(-1,1))
train_df.loc[:,'Close'] = scaler.fit_transform(train_df.Close.values.reshape(-1,1))
train_df.loc[:,'Dol'] = scaler.fit_transform(train_df.Dol.values.reshape(-1,1))

valid_df.loc[:,'Open'] = scaler.fit_transform(valid_df.Open.values.reshape(-1,1))
valid_df.loc[:,'High'] = scaler.fit_transform(valid_df.High.values.reshape(-1,1))
valid_df.loc[:,'Low'] = scaler.fit_transform(valid_df.Low.values.reshape(-1,1))
valid_df.loc[:,'Close'] = scaler.fit_transform(valid_df.Close.values.reshape(-1,1))
valid_df.loc[:,'Dol'] = scaler.fit_transform(valid_df.Dol.values.reshape(-1,1))

test_df.loc[:,'Open'] = scaler.transform(test_df.Open.values.reshape(-1,1))
test_df.loc[:,'High'] = scaler.transform(test_df.High.values.reshape(-1,1))
test_df.loc[:,'Low'] = scaler.transform(test_df.Low.values.reshape(-1,1))
test_df.loc[:,'Close'] = scaler.transform(test_df.Close.values.reshape(-1,1))
test_df.loc[:,'Dol'] = scaler.transform(test_df.Dol.values.reshape(-1,1))

frames = [train_df,valid_df,test_df]
data_raw = pd.concat(frames).as_matrix()

data = []

# create all possible sequences of length seq_len
for index in range(len(data_raw) - seq_len): 
      data.append(data_raw[index: index + seq_len])


data = np.array(data)

x_train = data[:train_set_size,:-1,:]
y_train = data[:train_set_size,-1,:]

x_valid = data[train_set_size:train_set_size+valid_set_size,:-1,:]
y_valid = data[train_set_size:train_set_size+valid_set_size,-1,:]

x_test = data[train_set_size+valid_set_size:,:-1,:]
y_test = data[train_set_size+valid_set_size:,-1,:]

cols = list(df.columns.values)

index_in_epoch = 0;
perm_array  = np.arange(x_train.shape[0])
np.random.shuffle(perm_array)

# function to get the next batch
def get_next_batch(batch_size):
    global index_in_epoch, x_train, perm_array   
    start = index_in_epoch
    index_in_epoch += batch_size
    
    if index_in_epoch > x_train.shape[0]:
        np.random.shuffle(perm_array) # shuffle permutation array
        start = 0 # start next epoch
        index_in_epoch = batch_size
        
    end = index_in_epoch
    return x_train[perm_array[start:end]], y_train[perm_array[start:end]]



n_steps = seq_len-1
n_inputs = 5
n_neurons = 200
n_outputs = 5
n_layers = 2
learning_rate = 0.001
batch_size = 50
n_epochs = 50        ############################### MUDAR NUMERO DE EPOCHS ###############################
train_set_size = x_train.shape[0]
test_set_size = x_test.shape[0]

tf.reset_default_graph()

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_outputs])

# Basic RNN Cell
# layers = [tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, activation=tf.nn.elu)
#           for layer in range(n_layers)]

# Basic LSTM Cell 
#layers = [tf.contrib.rnn.BasicLSTMCell(num_units=n_neurons, activation=tf.nn.elu)
#          for layer in range(n_layers)]

# GRU Cell
layers = [tf.contrib.rnn.GRUCell(num_units=n_neurons, activation=tf.nn.leaky_relu)
        for layer in range(n_layers)]

# Leaky LSTM Cell
# layers = [tf.contrib.rnn.LSTMCell(num_units=n_neurons,activation=tf.nn.leaky_relu, use_peepholes = True)
#          for layer in range(n_layers)]


multi_layer_cell = tf.contrib.rnn.MultiRNNCell(layers)
rnn_outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32)
stacked_rnn_outputs = tf.reshape(rnn_outputs, [-1, n_neurons]) 
stacked_outputs = tf.layers.dense(stacked_rnn_outputs, n_outputs)
outputs = tf.reshape(stacked_outputs, [-1, n_steps, n_outputs])
outputs = outputs[:,n_steps-1,:] # keep only last output of sequence
                                              
loss = tf.reduce_mean(tf.square(outputs[:,3] - y[:,3])) # loss function = mean squared error 
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate) 
training_op = optimizer.minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for iteration in range(int(n_epochs*train_set_size/batch_size)):
        x_batch, y_batch = get_next_batch(batch_size) # fetch the next training batch 
        sess.run(training_op, feed_dict={X: x_batch, y: y_batch}) 
        if iteration % int(5*train_set_size/batch_size) == 0:
            mse_train = loss.eval(feed_dict={X: x_train, y: y_train}) 
            mse_valid = loss.eval(feed_dict={X: x_valid, y: y_valid}) 
            print('%.2f epochs: MSE train/valid = %.6f/%.6f'%(
                iteration*batch_size/train_set_size, mse_train, mse_valid))

    y_train_pred = sess.run(outputs, feed_dict={X: x_train})
    y_valid_pred = sess.run(outputs, feed_dict={X: x_valid})
    y_test_pred = sess.run(outputs, feed_dict={X: x_test})
    


corr_price_development_train = np.sum(np.equal(np.sign(y_train_pred[1:,3]-y_train_pred[:-1,3]),
            np.sign(y_train[1:,3]-y_train[:-1,3])).astype(int)) / (y_train.shape[0]-1)
corr_price_development_valid = np.sum(np.equal(np.sign(y_valid_pred[1:,3]-y_valid_pred[:-1,3]),
            np.sign(y_valid[1:,3]-y_valid[:-1,3])).astype(int)) / (y_valid.shape[0]-1)
corr_price_development_test = np.sum(np.equal(np.sign(y_test_pred[1:,3]-y_test_pred[:-1,3]),
			np.sign(y_test[1:,3]-y_test[:-1,3])).astype(int)) / (y_test.shape[0]-1)

print('correct sign prediction for close(t) - close(t-1) price for train/valid/test: %.2f/%.2f/%.2f'%(
    corr_price_development_train, corr_price_development_valid, corr_price_development_test))


plt.figure(1)
plt.subplot(211)
plt.plot(y_test[:,3])
plt.title('Close')

plt.subplot(212)
plt.plot(y_test_pred[:,3])
plt.title('Predicted Close')

plt.figure(2)
plt.plot(y_test[:,3])
plt.plot(y_test_pred[:,3])

plt.show()


# a = pd.DataFrame(y_test_pred)
# b = pd.DataFrame(y_test)

# from pandas import ExcelWriter

# writer = ExcelWriter('resultados.xlsx')
# df.to_excel(writer,'df')
# a.to_excel(writer,'y_test_pred')
# b.to_excel(writer,'y_test')
# writer.save()
