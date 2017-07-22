"""
Train convolutional network for sentiment analysis. Based on
"Convolutional Neural Networks for Sentence Classification" by Yoon Kim
http://arxiv.org/pdf/1408.5882v2.pdf

For 'CNN-non-static' gets to 82.1% after 24 epochs with following settings:
embedding_dim = 20          
filter_sizes = (3, 4)
num_filters = 3
dropout_prob = (0.25, 0.5)
hidden_dims = 100

For 'CNN-rand' gets to 78-79% after 7-8 epochs with following settings:
embedding_dim = 20          
filter_sizes = (3, 4)
num_filters = 150
dropout_prob = (0.25, 0.5)
hidden_dims = 150

For 'CNN-static' gets to 75.4% after 7 epochs with following settings:
embedding_dim = 100          
filter_sizes = (3, 4)

num_filters = 150
dropout_prob = (0.25, 0.5)
hidden_dims = 150

* it turns out that such a small data set as "Movie reviews with one
sentence per review"  (Pang and Lee, 2005) requires much smaller network
than the one introduced in the original article:
- embedding dimension is only 20 (instead of 300; 'CNN-static' still requires ~100)
- 2 filter sizes (instead of 3)
- higher dropout probabilities and
- 3 filters per filter size is enough for 'CNN-non-static' (instead of 100)
- embedding initialization does not require prebuilt Google Word2Vec data.
Training Word2Vec on the same "Movie reviews" data set is enough to 
achieve performance reported in the article (81.6%)

** Another distinct difference is slidind MaxPooling window of length=2
instead of MaxPooling over whole feature map as in the article
"""
# tensorboard --logdir /Users/yy/PycharmProjects/CNN_text_classification_prevocab/logs

from __future__ import print_function
import numpy as np
import data_helpers
from w2v import train_word2vec
from keras.models import load_model
from keras.models import Sequential, Model
from keras.layers import Activation, Dense, Dropout, Embedding, Flatten, Input, Merge, Convolution1D, MaxPooling1D
from keras.optimizers import SGD
# from time import gmtime, strftime
import pickle
import objgraph
import keras
from keras import callbacks
from keras.callbacks import ModelCheckpoint
# from keras.callbacks import TensorBoard
#from memory_profiler import profile

# tb = TensorBoard(log_dir='./logs', histogram_freq=1)

# print(objgraph.show_most_common_types())
# from keras.models import model_fromjoson
import data_helpers as dh
import gensim
from random import shuffle
import keras.backend as K
from keras.callbacks import History
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from keras.utils import plot_model


import callbacks
callbacks.Callback()
history = History()


# aa = np.load('y_pre.npy')

# tb_cb = keras.callbacks.TensorBoard(log_dir='./data/', histogram_freq=1)
# cbks = [tb_cb]
# dh.download_images()
# np.random.seed(200)

# Parameters
# ==================================================
#
# Model Variations. See Kim Yoon's Convolutional Neural Networks for 
# Sentence Classification, Section 3 for detail.

# model_variation = 'CNN-static' #' CNN-non-static  #  CNN-rand | CNN-non-static | CNN-static
# print('Model variation is %s' % model_variation)

# Model Hyperparametersr
# sequence_length = 5000
embedding_dim = 50 #50
filter_sizes = (5, 5)
num_filters = 5
dropout_prob = (0.25, 0.5)
#dropout_prob = (0, 0)
hidden_dims = 100

# Training parameters
batch_size = 64#256
num_epochs = 200#100##0#0#0#1000#000#3000#00#3000
val_split = 0.1

# Word2Vec parameters, see train_word2vec
min_word_count = 1  # Minimum word count
context = 10        # Context window size

# Data Preparatopn
# ==================================================
#
# Load data
print("Loading data...")
# data_helpers.text_statistic()
# data_helpers.download_images('data/fake_or_real_news.csv','data/fake_or_real_news_images')
x_text, x_arti_text, x_image, x_arti_images, y, vocabulary, vocabulary_inv,max_len = data_helpers.load_all_data()
# np.save('data/y.npy',y)
# x = np.load('data/x.npy')
# y = np.load('data/y.npy')
seq_length = max_len
all_data = [x_text, x_arti_text, x_image, x_arti_images, y]
# print("imalanced data preparation")
# all_data = data_helpers.imbalanced_data(all_data)
# print('imbalanced data complete')
# np.save('data/x_text1_artificial.npy',x_arti_text)
# np.save('data/x_text_feature.npy',x_text)
# pickle.dump(vocabulary, open("data/vocab/vocabulary.p", "wb"))
# pickle.dump(vocabulary_inv, open("data/vocab/vocabulary_inv.p", "wb"))
# vocabulary = pickle.load(open("data/vocab/vocabulary.p",'rb'))
# vocabulary_inv = pickle.load(open("data/vocab/vocabulary_inv.p",'rb'))


# print("all_train_data_num:",len(shuffled_data[0]),"train_pos_num:",np.asarray(shuffled_data[-1]).tolist().count([1,0]))

partial_data = all_data #[data[:] for data in shuffled_data]

train_percent = 0.8

data_num = len(partial_data[0])
train_num = int(0.8*data_num)
train_data = [data[:train_num] for data in partial_data]
validate_index = train_num + int(0.5*(data_num-train_num))
validate_data = [data[train_num:validate_index] for data in partial_data]
test_data = [data[validate_index:] for data in partial_data]


# sample _weights = [2.0 if label == [1, 0] else 1.0 for label in all_data[-1]]

print("train_data_num:",len(train_data[0]),"train_pos_num:",np.asarray(train_data[-1]).tolist().count([1,0]))
print("validate_data_num:",len(validate_data[0]),"validate_pos_num:",np.asarray(validate_data[-1]).tolist().count([1,0]))
print("test_data_num:",len(test_data[0]),"test_pos_num:",np.asarray(test_data[-1]).tolist().count([1,0]))


# x_text = np.load('data/x_text_feature.npy')
# x_text_artificial = np.load('data/x_text_artificial.npy')
# x_image = np.load('data/images.npy')
# x_image = np.random.randint(100,size=(18666,3,100,100))
# x_image_artificial = np.load('data/x_image_artificial.npy')
# x_text = x_text[:18666]
# y = y[:18666]
#
# # overall_model = data_helpers.build_model(x_text, x_text_artificial,x_image,x_image_artificial)
# if model_variation=='CNN-non-static' or model_variation=='CNN-static':
#     embedding_weights = train_word2vec(x_text, vocabulary_inv, embedding_dim, min_word_count, context)
#     if model_variation=='CNN-static':
#         x = embedding_weights[0][x_text]
# elif model_variation=='CNN-rand':
#     embedding_weights = None
# else:
#     raise ValueEriror('Unknown model variation')

# Shuffle data
# shuffle_indices = np.random.permutation(np.arange(len(y)))
# x_shuffled = x[shuffle_indices]
# y_shuffled = y[shuffle_indices].argmax(axis=1)
#
# x_train = x_shuffled[:int(len(y)*2/3)]
# y_train = y_shuffled[:int(len(y)*2/3)]
# x_test =  x_shuffled[int(len(y)*2/3):]`
# y_test =  y_shuffled[int(len(y)*2/3):]

# print("Vocabulary Size: {:d}".format(len(vocabulary)))

print('BUild model')
overall_model = data_helpers.build_model(train_data,vocabulary,vocabulary_inv,seq_length)
print(overall_model.summary())
# plot_model(overall_model, to_file='model.png')

# overall_model = data_helpers.build_text_model(train_data,vocabulary,vocabulary_inv,seq_length)
opt = keras.optimizers.RMSprop(lr=1e-01,rho=0.9, epsilon=1e-08, decay=0.0) # nice result too
tsb = callbacks.EarlyStoppingByLossVal(monitor='loss', value=0.8, verbose=1)

# opt = keras.optimizers.RMSprop(lr=1,rho=0.9, epsilon=1e-08, decay=0.0) # nice result too

# opt = keras.optimizers.RMSprop(lr=1e-03, rho=0.9, epsilon=1e-08, decay=0.5)
# opt = keras.optimizers.SGD(lr=1e-4, momentum=0.9, decay=0.0, nesterov=True)#, clipnorm=1.)
# opt = keras.optimizers.Adam(lr=1e-5)#, beta_1=0.9, beta_2=0.999, epsilon=1e-08)#, decay=0.1)
# opt = keras.optimizers.Adagrad(lr=1e-1, epsilon=1e-08, decay=0.0) # best so far
# opt = keras.optimizers.Adagrad(lr=1e-3, epsilon=1e-08, decay=0.0) # small learning rate


def f1score(y_true, y_pred):
    num_tp = K.sum(y_true*y_pred)
    num_fn = K.sum(y_true*(1.0-y_pred))
    num_fp = K.sum((1.0-y_true)*y_pred)
    num_tn = K.sum((1.0-y_true)*(1.0-y_pred))
    #print num_tp, num_fn, num_fp, num_tn
    if 2.0*num_tp+num_fn+num_fp == 0:
        return 0
    f1 = 2.0*num_tp/(2.0*num_tp+num_fn+num_fp)

    # print("f1:",f1)
    return f1

def precision(y_true, y_pred):
    num_tp = K.sum(y_true*y_pred)
    num_fn = K.sum(y_true*(1.0-y_pred))
    num_fp = K.sum((1.0-y_true)*y_pred)
    num_tn = K.sum((1.0-y_true)*(1.0-y_pred))
    #print num_tp, num_fn, num_fp, num_tn
    if num_tp+num_fp == 0:
        return 0
    precision_val = num_tp/(num_tp+num_fp)

    # print("precision:",precision_val)
    return precision_val

def recall(y_true, y_pred):
    num_tp = K.sum(y_true*y_pred)
    num_fn = K.sum(y_true*(1.0-y_pred))
    num_fp = K.sum((1.0-y_true)*y_pred)
    num_tn = K.sum((1.0-y_true)*(1.0-y_pred))
    #print num_tp, num_fn, num_fp, num_tn
    if num_tp+num_fn == 0:
        return 0
    recall_val = num_tp/(num_tp+num_fn)

    # print("recall:",recall_val)
    return recall_val

overall_model.compile(optimizer=opt,
                      loss='binary_crossentropy',
                      # metrics=['binary_accuracy', f1score], clipnorm=1.)#, 'fmeasure', 'precision', 'recall'])
                      metrics=['binary_accuracy', f1score, precision, recall], clipnorm=1.)#,
                      #sample_weight_mode=)#, 'fmeasure', 'precision', 'recall'])
# Building model
# ==================================================
#
# graph subnet with one input and one output,
# convolutional layers concateneted in c
model = overall_model
# model.summary()
# Training model
# # ==================================================
# model.fit([x_text,x_arti_text,x_image,x_arti_images], y, batch_size=batch_size,
#         nb_epoch=num_epochs, validation_split=val_split, verbose=2)



#@threadsafe_generator



generator_batch = data_helpers.batch_iter((train_data[0:4],train_data[-1]), batch_size=batch_size,
                                            num_epochs=num_epochs, max_pad_seq_len=max_len)

val_generator_batch = data_helpers.batch_iter((validate_data[0:4],validate_data[-1]), batch_size=batch_size,
                                            num_epochs=num_epochs, max_pad_seq_len=max_len)

# tsb = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)
# tsb = keras.callbacks.EarlyStopping(monitor='loss', min_delta=1e-4, patience=5, verbose=1, mode='min')
# tsb = keras.callbacks.EarlyStopping(monitor='loss', min_delta=1e-4, patience=5, verbose=1, mode='min')
# tsb = callbacks.EarlyStoppingByAcc(monitor='val_acc', value=0.91, verbose=1)


# remote = callbacks.RemoteMonitor(root='http://127.0.0.1:6006',path='./logs',
#                                 field = 'data', headers = None)

model.fit_generator(generator_batch,
                    samples_per_epoch = 64,#1024,
                    nb_epoch = num_epochs,
                    #show_accuracy = True,
                    validation_data = val_generator_batch,#generator_batch,#threadsafe_generator(generator_batch),
                    nb_val_samples = int(len(validate_data[0])),
                    max_q_size = 10,
                    callbacks = [tsb],
                    # class_weight = {0:0.01, 1:0.99},
                    class_weight={0: 5, 1: 3},
                    verbose=2)

# get weights
test_generator_batch = data_helpers.batch_iter((test_data[0:4],test_data[-1]), batch_size=batch_size,
                                                num_epochs=num_epochs, max_pad_seq_len=max_len)
score = model.evaluate_generator(test_generator_batch, 1, max_q_size=10, workers=1, pickle_safe=False)#, steps=int(len(test_data[0])*0.1), max_q_size=10, workers=1, pickle_safe=False)

print("Start to evaluate.")
score = model.evaluate_generator(test_generator_batch,steps=len(test_data[0])/batch_size)#,val_samples=len(test_data[0]),max_q_size=10,nb_worker=1, pickle_safe=False)
print("%s: %.2f%%" % (model.metrics_names[1], score[1] * 100))
print("%s: %.2f%%" % (model.metrics_names[2], score[2] * 100))
print("%s: %.2f%%" % (model.metrics_names[3], score[3] * 100))
print("%s: %.2f%%" % (model.metrics_names[4], score[4] * 100))
# model.save_weights("model.h5")
# model.save('my_model.h5')
# print("Saved model to disk")
# model = load_model('my_model.h5',custom_objects={'f1score': f1score,'precision': precision,'recall': recall})
# load json and create model
# del model
# json_file = open('model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# from keras.models import model_from_json
# model = model_from_json(loaded_model_json)
# load weights into new model
# model.load_weights("model.h5",custom_objects={'f1score': f1score})
# print("Loaded model from disk")
# y_pred = model.predict_generator(test_generator_batch, len(test_data[0]), max_q_size=10, nb_worker=1, pickle_safe=False)
# ypred = np.argmax(y_pred, axis=1)


train_data_features = data_helpers.generate_features(train_data,max_len)
y_predict = model.predict(train_data_features[0:4],batch_size=1,verbose=2)
np.save('y_pre.npy', y_predict)
ytrue = np.argmin(np.asarray(train_data_features[-1]),axis=1)
ypred = np.argmin(y_predict,axis=1)
# precision,recall,f1,true_sum = precision_recall_fscore_support(ytrue, ypred[:len(test_data[0])], average='macro')
precision,recall,f1,true_sum = precision_recall_fscore_support(ytrue, ypred)
print("Confusion matrix:\n",confusion_matrix(ytrue, ypred))
print('Precision:',precision,'Recall:',recall,'F1',f1)


test_data_fetures = data_helpers.generate_features(test_data,max_len)
y_predict = model.predict(test_data_fetures[0:4],batch_size=1,verbose=2)
np.save('y_pre.npy', y_predict)
ytrue = np.argmin(np.array(test_data_fetures[-1]),axis=1)
ypred = np.argmin(y_predict,axis=1)
# precision,recall,f1,true_sum = precision_recall_fscore_support(ytrue, ypred[:len(test_data[0])], average='macro')
precision,recall,f1,true_sum = precision_recall_fscore_support(ytrue, ypred)
print("Confusion matrix:\n",confusion_matrix(ytrue, ypred))
print('Precision:',precision,'Recall:',recall,'F1',f1)

# print(objgraph.show_most_common_types())

# predict_y = model.predict_classes([x_text,x_arti_text,x_image,x_arti_images])
# from sklearn.metrics import precision_recall_fscore_support
# print("predict value",predict_y)
# precision, recall, f1, _ = precision_recall_fscore_support(y,predict_y)
# print ('\n')
# print (precision,recall,f1)
# para_file = open('data/results.txt','a')
# para_file.write('\n'+'\n')
# para_file.write(strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime())+'\n')
# para_file.write('precision = '+str(precision)+'\n')
# para_file.write('recall = '+str(recall)+'\n')
# para_file.write('f1-measure = '+str(f1)+'\n')
# para_file.write('\n'+'\n')
# para_file.close()
#
# # Save model
# model.save('test.h5')
# model.save_weights('my_model_weights.h5')
# loaded_model = data_helpers.build_model(x_text,x_arti_text,x_image,x_arti_images,vocabulary,vocabulary_inv,seq_length)
# loaded_model.load_weights('test.h5')
# # loaded_model = load_model('test.h5')
# loaded_model.compile(optimizer='rmsprop',
#                             loss='binary_crossentropy',
#                             metrics=['binary_accuracy', 'fmeasure', 'precision', 'recall'])
# loaded_model.summary()
# # evaluate loaded model on test data
# score = loaded_model.evaluate([x_text,x_arti_text,x_image,x_arti_images],y, verbose=2)
#
# print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1] * 100))
# print("%s: %.2f%%" % (loaded_model.metrics_names[2], score[2] * 100))
# print("%s: %.2f%%" % (loaded_model.metrics_names[3], score[3] * 100))
# print("%s: %.2f%%" % (loaded_model.metrics_names[4], score[4] * 100))
#
#
# predict_y = loaded_model.predict_classes([x_text,x_arti_text,x_image,x_arti_images])
# from sklearn.metrics import precision_recall_fscore_support
# precision, recall, f1, _ = precision_recall_fscore_support(y,
#                                                     predict_y)
# from sklearn.metrics import confusion_matrix
# print(confusion_matrix(y, predict_y.reshape(1,len(list(y)))[0]))

# print('\n')
# print(precision,recall,f1)

# para_file = open('attModel/parameters.txt','a')
# para_file.write('\n'+'\n')
# para_file.write(strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime())+'\n')
# para_file.write('sequence_length = '+str(sequence_length)+'\n')
# para_file.write('embedding_dim = '+str(20)+'\n')
# para_file.write('filter_sizes = '+str(filter_sizes)+'\n')
# para_file.write('num_filters = '+str(num_filters)+'\n')
# para_file.write('dropout_prob = '+str(dropout_prob)+'\n')
# para_file.write('hidden_dims = '+str(hidden_dims)+'\n')
# para_file.write('batch_size = '+str(batch_size)+'\n')
# para_file.write('num_epochs = '+str(num_epochs)+'\n')
# para_file.write('val_split = '+str(val_split)+'\n')
# para_file.write('min_word_count = '+str(min_word_count)+'\n')
# para_file.write('context = '+ str(context)+'\n')
# para_file.write('\n'+'\n')
# para_file.close()
