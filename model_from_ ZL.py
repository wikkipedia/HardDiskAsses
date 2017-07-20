
from keras.models import Sequential,Model
from keras.layers import *
from keras.optimizers import *
import pickle
from utils import *
from yellowfin import YFOptimizer

from CHE import *
from keras.regularizers import *


def attention_3d_block(inputs,dense):
    attention = dense(inputs)

    attention = Flatten()(attention)

    attention = Activation('softmax')(attention)
    attention = RepeatVector(params['factor_dim'])(attention)

    attention = Permute([2, 1])(attention)

    # apply the attention
    sent_representation = merge([inputs, attention], mode='mul')
    sent_representation = Lambda(lambda xin: K.sum(xin, axis=1))(sent_representation)
    # print sent_representation.get_shape()
    # sent_representation = Flatten()(sent_representation)
    return sent_representation

def sum_hiddens(inputs):
    sent_representation = Lambda(lambda xin: K.mean(xin, axis=1))(inputs)
    return sent_representation


def build(n_users,n_items,max_doc_len,num_docs):

    #test
    word2id_file = open('word2id','rb')
    word2id = pickle.load(word2id_file)

    # build embedding layer
    word_dim = params['embedding_dim']  # dimensionality of your word vectors
    n_symbols = len(word2id) + 1  # adding 1 to account for 0th index (for masking)

    pos_target_doc_input = Input(shape=(max_doc_len,), dtype='int32',name='pos_doc_input')
    neg_target_doc_input = []
    for _ in xrange(params['num_neg_doc']):
        neg_target_doc_input.append(Input(shape=(max_doc_len,), dtype='int32'))



    embedding = Embedding(output_dim=word_dim, input_length=max_doc_len, input_dim=n_symbols,
                          embeddings_regularizer=l2(params['l2']),embeddings_initializer='he_normal')

    doc_lstm = GRU(params['factor_dim'],return_sequences=True,kernel_regularizer=l2(params['l2']),kernel_initializer='he_normal',implementation=2)

    dense = Dense(1, activation='tanh',kernel_regularizer=l2(params['l2']),kernel_initializer='he_normal', name='attention')




    target_doc_model = embedding(pos_target_doc_input)

    #target_doc_model = Dropout(0.2)(target_doc_model)
    target_doc_model = BatchNormalization()(target_doc_model)

    target_doc_model = doc_lstm(target_doc_model)
    #target_doc_model = Dropout(0.2)(target_doc_model)
    target_doc_model = BatchNormalization()(target_doc_model)
    #target_doc_model = Attention()(target_doc_model)
    if params['apply_attention']:
        target_doc_model = attention_3d_block(target_doc_model,dense=dense)
    else:
        target_doc_model = sum_hiddens(target_doc_model)


    neg_target_doc_model = []
    for i in xrange(params['num_neg_doc']):

        neg_target_doc_model1 = embedding(neg_target_doc_input[i])
        neg_target_doc_model1 = BatchNormalization()(neg_target_doc_model1)

        neg_target_doc_model1 = doc_lstm(neg_target_doc_model1)
        #neg_target_doc_model = Dropout(0.2)(neg_target_doc_model)
        neg_target_doc_model1 = BatchNormalization()(neg_target_doc_model1)
        if params['apply_attention']:
            neg_target_doc_model1 = attention_3d_block(neg_target_doc_model1,dense=dense)
        else:
            neg_target_doc_model1 = sum_hiddens(neg_target_doc_model1)
        #neg_target_doc_model1 = Attention()(neg_target_doc_model1)
        neg_target_doc_model.append(neg_target_doc_model1)


    user_id = Input(shape=(1,),dtype='float32',name='user_id')
    pos_doc_id = Input(shape=(1,),dtype='float32',name='pos_item_id')
    neg_doc_id =[]
    for _ in xrange(params['num_neg_doc']):
        neg_doc_id.append(Input(shape=(1,), dtype='float32'))


    main_output = merge([user_id,pos_doc_id]+neg_doc_id + [target_doc_model] + neg_target_doc_model, mode='concat')
    main_output = CHE(num_users=n_users,num_items=n_items,lambda_u=params['lambda_u']
                      ,lambda_v=params['lambda_v'],factor_dim=params['factor_dim'])(main_output)
    model = Model(input=[user_id,pos_doc_id] + neg_doc_id + [pos_target_doc_input] + neg_target_doc_input, output=[main_output])

    model.summary()

    def loss(y_true,y_pred): return K.mean(y_pred,axis=-1)

    model.compile(loss=loss, optimizer=Adam(lr=params['lr']))#,loss_weights={'che_1': 1,'prediction': 0.0})
    return model
