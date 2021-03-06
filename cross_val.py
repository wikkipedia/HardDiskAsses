# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 10:14:56 2017

@author: WIKKI
"""

import gc
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
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
from keras.wrappers.scikit_learn import KerasClassifier


def load_data(file_path, params):
    Y16 = pd.read_pickle(file_path)
    dr = ['smart_183_normalized','smart_183_raw','date','failure','left_day']
    npdata = Y16.drop(dr,axis=1)
    npdata = np.array(npdata)
    del Y16
    
    #feature normalization
    temp = npdata[:,1:-1]        
    min_max_scaler = preprocessing.MinMaxScaler()
    temp = min_max_scaler.fit_transform(temp)
    npdata[:,1:-1] = temp
    
    #find the last log idx of each hard disk
    serial = []
    idx = []
    pre = 'begin'
    l = len(npdata)
    for i in range(0, l):
        if npdata[i,0] != pre:
            serial.append(pre)
            idx.append(i-1)
            pre = npdata[i,0]
    idx.append(i)
    serial.append(pre)
    last_idx = dict(zip(serial,idx))
    
    #create the data set with the sequence window = seq_window
    now = 'begin'
    x_data = []
    y_data = []
    for i in range(0,l):
        if npdata[i,0] != now:
            now = npdata[i,0]
            last = last_idx[now]
        if npdata[i,-1]<7:
            end = min(i+params['seq_window']-1, last) + 1
            sliced = npdata[i:end,1:-1]
            x_data.append(sliced)
            y_data.append(npdata[i,-1])
    del npdata
    
    #padding with zeros
    data = x_data
    for i in range(0,len(data)):
        result = data[i]
        result = np.pad(result, ((0,params['seq_window']-result.shape[0]),(0,0)), 'constant', constant_values=0)
        data[i] = result
    x_data = np.array(data)
    x_data = x_data.astype(np.float32)
    y_data = [c - 1 for c in y_data]  #make the label start from 0
    y_data = np.asarray(y_data) #for cross validation
    del data
    gc.collect()
    return x_data, y_data


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
    # print sent_representation.get_shape()
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
          'nb_echop': 2500}
seed = 7
np.random.seed(seed)

feature_data, label_data = load_data('./Y2', params)

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
cvscores = []
clf_score = []
for train, test in kfold.split(feature_data, label_data):
    model = build_model(params)

    #encode outputs
    label_train = np_utils.to_categorical(label_data[train].tolist(), num_classes=params['n_classes'])
    label_test = np_utils.to_categorical(label_data[test].tolist(), num_classes=params['n_classes'])

    model.fit(feature_data[train], label_train, batch_size=params['batch_size'], nb_epoch=params['nb_echop'], verbose=2)
    scores = model.evaluate(feature_data[test], label_test, batch_size=params['batch_size'], verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1] * 100)
    label_pred = model.predict(feature_data[test], batch_size=params['batch_size'], verbose=0)
    clf_score.append(clf_evaluate(label_test, label_pred))
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
clf_score = np.asarray(clf_score)
result = np.mean(clf_score, axis=0)
print(result)
