# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 10:14:56 2017

@author: WIKKI
"""

import gc
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
import keras
import keras.callbacks
from keras.utils import np_utils
from keras import backend as K
from keras.models import *
from keras.optimizers import *
from keras.utils import *
from keras.regularizers import *
from keras.layers import *
from keras.layers.core import *
from keras.layers.recurrent import *
from keras.layers.normalization import *


def load_data(file_path, params):
    Y16 = pd.read_pickle(file_path)
    dr = ['smart_183_normalized', 'smart_183_raw', 'date', 'failure']
    npdata = Y16.drop(dr, axis=1)
    npdata = np.array(npdata)
    del Y16

    # feature normalization
    temp = npdata[:, 1:-2]
    min_max_scaler = preprocessing.MinMaxScaler()
    temp = min_max_scaler.fit_transform(temp)
    npdata[:, 1:-2] = temp

    # find the last log idx of each hard disk
    serial = []
    idx = []
    pre = 'begin'
    l = len(npdata)
    for i in range(0, l):
        if npdata[i, 0] != pre:
            serial.append(pre)
            idx.append(i - 1)
            pre = npdata[i, 0]
    idx.append(i)
    serial.append(pre)
    last_idx = dict(zip(serial, idx))

    # create the data set with the sequence window = seq_window
    now = 'begin'
    x_data = []
    other_data = []
    serial_data = []
    for i in range(0, l):
        if npdata[i, 0] != now:
            now = npdata[i, 0]
            last = last_idx[now]
        if npdata[i, -1] < 7:
            end = min(i + params['seq_window'] - 1, last) + 1
            sliced = npdata[i:end, 1:-2]
            x_data.append(sliced)
            serial_data.append(npdata[i, 0])
            other_data.append(npdata[i, -2:])
    del npdata
    serial_data = np.asarray(serial_data)
    serial_data = serial_data.reshape(serial_data.shape[0], 1)
    other_data = np.concatenate((serial_data, other_data), axis=1)

    # padding with zeros
    data = x_data
    for i in range(0, len(data)):
        result = data[i]
        result = np.pad(result, ((0, params['seq_window'] - result.shape[0]), (0, 0)), 'constant', constant_values=0)
        data[i] = result
    x_data = np.array(data)
    del data

    # split train/test data
    x_data = x_data.astype(np.float32)
    x_train, x_test, y_train, y_test = train_test_split(x_data, other_data, test_size=0.1, random_state=42)
    gc.collect()

    # create idx and labels
    train_label = y_train[:, -1].tolist()
    test_idx = y_test[:, 0:2]
    test_label = y_test[:, -1].tolist()

    # make the label start from 0
    train_label = [c - 1 for c in train_label]
    test_label = [d - 1 for d in test_label]

    # encode outputs
    train_label = np_utils.to_categorical(train_label, num_classes=params['n_classes'])
    test_label = np_utils.to_categorical(test_label, num_classes=params['n_classes'])

    return x_train, x_test, train_label, test_label, test_idx


def attention_layer(inputs, layer):
    attention = layer(inputs)
    attention = Dense(1, use_bias=False)(attention)
    attention = Flatten()(attention)
    attention = Activation('softmax')(attention)
    attention = RepeatVector(params['rnn_dim'])(attention)
    attention = Permute([2, 1])(attention)

    # apply the attention
    sent_representation = merge([inputs, attention], mode='mul')
    sent_representation = Lambda(lambda xin: K.sum(xin, axis=1))(sent_representation)
    return sent_representation


def sum_hidden(inputs):
    sent_representation = Lambda(lambda xin: K.mean(xin, axis=1))(inputs)
    return sent_representation


def build_model(params):
    #rnn_layer = GRU(params['rnn_dim'], return_sequences=True, kernel_regularizer=l2(params['l2']), kernel_initializer='he_normal', implementation=2)
    rnn_layer = GRU(params['rnn_dim'], return_sequences=True)
    #rnn_layer = Bidirectional(GRU(params['rnn_dim'], return_sequences=True))
    #rnn_layer = SimpleRNN(params['rnn_dim'], return_sequences=True)

    #dense = Dense(params['rnn_dim'], activation='tanh', kernel_regularizer=l2(params['l2']), kernel_initializer='he_normal')
    dense = Dense(params['rnn_dim'], activation='tanh')

    model_input = Input(shape=(params['seq_window'],params['feature_num']), dtype='float32', name='main_input')
    rnn_output = rnn_layer(model_input)
    rnn_output = BatchNormalization()(rnn_output)
    #rnn_output = Dropout(params['dropout'])(rnn_output)

    if params['apply_attention']:
        sub_output = attention_layer(rnn_output, layer=dense)
    else:
        sub_output = sum_hidden(rnn_output)

    health_status = Dense(params['n_classes'], activation='softmax', name='main_output')(sub_output)

    model = Model(inputs=model_input, outputs=health_status)
#    model.summary()

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def clf_evaluate(y_true, y_pred):
    y_true = np.argmax(y_true, axis=1)
    y_true = y_true.tolist()
    y_pred = np.argmax(y_pred,axis=1)
    y_pred = y_pred.tolist()
    r = metrics.precision_score(y_true, y_pred, average=None).tolist()
    r = r + metrics.recall_score(y_true, y_pred, average=None).tolist()
    r = r + metrics.f1_score(y_true, y_pred, average=None).tolist()
    r.append(metrics.precision_score(y_true, y_pred, average='macro'))
    r.append(metrics.recall_score(y_true, y_pred, average='macro'))
    r.append(metrics.f1_score(y_true, y_pred, average='macro'))
    return r

params = {'seq_window': 20,
          'feature_num': 33,
          'apply_attention': True,
          'l2': 0.01,
          'dropout': 0.1,
          'rnn_dim': 64,
          'n_classes': 6,
          'batch_size': 512,
          'nb_echop': 100,
          'repeat_num': 3}

train_data, test_data, train_label, test_label, test_idx = load_data('./Y2', params)

cvscores = []
clf_score = []
max_acc = 0
for repeat in range(params['repeat_num']):
    #np.random.seed(repeat)
    model = build_model(params)
    model.fit(train_data, train_label, batch_size=params['batch_size'], nb_epoch=params['nb_echop'], verbose = 2)
    scores = model.evaluate(test_data, test_label, batch_size = params['batch_size'], verbose = 0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1] * 100)
    label_pred = model.predict(test_data, batch_size=params['batch_size'], verbose=0)
    if scores[1]>max_acc:
        max_acc = scores[1]
        best_pred = label_pred
    clf_score.append(clf_evaluate(test_label, label_pred))
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
clf_score = np.asarray(clf_score)
result = np.mean(clf_score, axis=0)
print(result)
np.save("best_output.npy", best_pred)
best_pred = np.argmax(best_pred, axis=1)
best_pred = best_pred.reshape(best_pred.shape[0],1)
pred_output = np.concatenate((test_idx, best_pred), axis=1)
np.save("pred_output.npy", pred_output)
