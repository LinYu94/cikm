import re

from keras import backend as K
from keras.layers import Layer
from keras.preprocessing.sequence import pad_sequences
from keras import initializers, regularizers, constraints

from nltk.corpus import stopwords
from gensim.models import KeyedVectors
import tensorflow as tf
from config_dev import Config
import gensim
import keras
from keras.models import Model, Sequential
from keras.layers import Input, Embedding, LSTM, GRU, Conv1D, Conv2D, GlobalMaxPool1D, Dense, Dropout, Bidirectional
from keras.layers import BatchNormalization

import numpy as np
import itertools
import pandas as pd

config = Config()

class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        """
        Keras Layer that implements an Attention mechanism for temporal data.
        Supports Masking.
        Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Example:
            model.add(LSTM(64, return_sequences=True))
            model.add(Attention())
        """
        self.supports_masking = True
        #self.init = initializations.get('glorot_uniform')
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        # eij = K.dot(x, self.W) TF backend doesn't support it

        # features_dim = self.W.shape[0]
        # step_dim = x._keras_shape[1]

        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())
        print(a)
        # in some cases especially in the early stages of training the sum may be almost zero
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
    #print weigthted_input.shape
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        #return input_shape[0], input_shape[-1]
        return input_shape[0],  self.features_dim
     
class Attention2(Layer):
    def __init__(self, step_dim, **kwargs):
        """
        Keras Layer that implements an Attention mechanism for temporal data.
        Supports Masking.
        Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Example:
            model.add(LSTM(64, return_sequences=True))
            model.add(Attention())
        """
        self.supports_masking = True
        #self.init = initializations.get('glorot_uniform')
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention2, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.features_dim = input_shape[-1]
        self.built = True

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        # eij = K.dot(x, self.W) TF backend doesn't support it
        # features_dim = self.W.shape[0]
        # step_dim = x._keras_shape[1]

        features_dim = self.features_dim
        step_dim = self.step_dim
        # print(x.shape)
        last_hidden = x[0][-1]
        print('<' * 30)
        print(last_hidden.shape)
        print('features_dim: ', features_dim)
        # eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))), (-1, step_dim))
        ij = K.dot(K.reshape(x, (-1, features_dim)), K.reshape(last_hidden, (features_dim, 1)))
        print('*' * 30)
        print(ij)
        eij = K.reshape(ij, (-1, step_dim))
        
        eij = K.tanh(eij)

        a = K.exp(eij)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())
        print(a)
        # in some cases especially in the early stages of training the sum may be almost zero
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
    #print weigthted_input.shape
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        #return input_shape[0], input_shape[-1]
        return input_shape[0],  self.features_dim
   
     
def BuildESModel(embeddings, embedding_dim):
    # --
    # Load EN model
    print('Loading EN encoder...: ', config.en_bst_model_path)
    enModel = keras.models.load_model(config.en_bst_model_path, custom_objects={'ManDist': ManDist, 'Attention': Attention})
    enEncoder = enModel.layers[2]
    print(enEncoder)
    for layer in enEncoder.layers:
        layer.trainable = False
    
    print(enEncoder.summary())

    # Define the shared model
    x = Sequential()
    x.add(Embedding(len(embeddings), embedding_dim,
                    weights=[embeddings], 
                    input_shape=(config.es_max_seq_length,),
                    trainable=False))
    
    # LSTM
    x.add(Bidirectional(LSTM(config.n_hidden,
                            dropout=config.dropout_rate, 
                            recurrent_dropout=config.dropout_rate), name='es_encoder'))

    # x.add(Bidirectional(LSTM(config.n_hidden, 
    #                         dropout=config.dropout_rate, 
    #                         recurrent_dropout=config.dropout_rate)))

    x.add(Attention(config.en_max_seq_length))
    shared_model = x

    # The visible layer
    en_input = Input(shape=(config.en_max_seq_length,), dtype='int32')

    es_left_input = Input(shape=(config.es_max_seq_length,), dtype='int32')
    es_right_input = Input(shape=(config.es_max_seq_length,), dtype='int32')

    # Pack it all up into a Manhattan Distance model
    # malstm_distance = ManDist()([shared_model(left_input), shared_model(right_input)])

    ss_malstm_distance = ManDist()([shared_model(es_left_input), shared_model(es_right_input)])
    sn1_malstm_distance = ManDist()([shared_model(es_left_input), enEncoder(en_input)])
    sn2_malstm_distance = ManDist()([shared_model(es_right_input), enEncoder(en_input)])

    model = Model(inputs=[es_left_input, es_right_input, en_input], 
                outputs=[ss_malstm_distance, sn1_malstm_distance, sn2_malstm_distance])

    # `multi_gpu_model()` is a so quite buggy. it breaks the saved model.
    if config.gpus >= 2:
        model = keras.utils.multi_gpu_model(model, gpus=config.gpus)

    model.compile(loss='binary_crossentropy', 
                  optimizer=keras.optimizers.Adam(),
                  loss_weights=[1, 0.4, 0.4],
                  metrics=['accuracy'])
    model.summary()
    shared_model.summary()
    return model

def BuildESModel2(embeddings, embedding_dim):
    # --
    # Load EN model
    print('Loading EN encoder...: ', config.en_bst_model_path)
    enModel = keras.models.load_model(config.en_bst_model_path, custom_objects={'Euclidean': Euclidean})
    enEncoder = enModel.layers[2]
    print(enEncoder)
    for layer in enEncoder.layers:
        layer.trainable = False
    
    print(enEncoder.summary())

    # Define the shared model
    x = Sequential()
    x.add(Embedding(len(embeddings), embedding_dim,
                    weights=[embeddings], 
                    input_shape=(config.es_max_seq_length,),
                    trainable=False))
    
    # LSTM
    x.add(Bidirectional(LSTM(config.n_hidden,
                            dropout=config.dropout_rate, 
                            recurrent_dropout=config.dropout_rate), name='es_encoder'))

    # x.add(Bidirectional(LSTM(config.n_hidden, 
    #                         dropout=config.dropout_rate, 
    #                         recurrent_dropout=config.dropout_rate)))
    shared_model = x

    # The visible layer
    en_input = Input(shape=(config.en_max_seq_length,), dtype='int32')
    en_input_same = Input(shape=(config.en_max_seq_length,), dtype='int32')

    es_left_input = Input(shape=(config.es_max_seq_length,), dtype='int32')
    es_right_input = Input(shape=(config.es_max_seq_length,), dtype='int32')

    # Pack it all up into a Manhattan Distance model
    # malstm_distance = ManDist()([shared_model(left_input), shared_model(right_input)])

    ss_malstm_distance = Euclidean()([shared_model(es_left_input), shared_model(es_right_input)])
    sn1_malstm_distance = Euclidean()([shared_model(es_left_input), enEncoder(en_input)])
    sn2_malstm_distance = Euclidean()([shared_model(es_right_input), enEncoder(en_input_same)])

    model = Model(inputs=[es_left_input, es_right_input, en_input, en_input_same], 
                outputs=[ss_malstm_distance, sn1_malstm_distance, sn2_malstm_distance])

    # `multi_gpu_model()` is a so quite buggy. it breaks the saved model.
    if config.gpus >= 2:
        model = keras.utils.multi_gpu_model(model, gpus=config.gpus)

    model.compile(loss='binary_crossentropy', 
                  optimizer=keras.optimizers.Adam(),
                #   loss_weights=[1, 0.8, 0.8],
                  metrics=['accuracy'])
    print(model.summary())
    shared_model.summary()
    return model

def BuildENModel(embeddings, embedding_dim):
    # --
    # Define the shared model
    x = Sequential()
    x.add(Embedding(len(embeddings), 
                    embedding_dim,
                    weights=[embeddings], 
                    input_shape=(config.en_max_seq_length,), 
                    trainable=False))

    # LSTM
    x.add(Bidirectional(LSTM(config.n_hidden,
                            return_sequences=True,
                            dropout=config.dropout_rate, 
                            recurrent_dropout=config.dropout_rate), name='en_encoder'))
    # x.add(Dropout(0.2))
    x.add(Attention2(config.en_max_seq_length))
    
    shared_model = x
    
    # The visible layer
    left_input = Input(shape=(config.en_max_seq_length,), dtype='int32')
    right_input = Input(shape=(config.en_max_seq_length,), dtype='int32')

    # Pack it all up into a Manhattan Distance model
    # distance = ManDist()([shared_model(left_input), shared_model(right_input)])
    distance = Euclidean()([shared_model(left_input), shared_model(right_input)])

    model = Model(inputs=[left_input, right_input], 
                outputs=[distance])

    # `multi_gpu_model()` is a so quite buggy. it breaks the saved model.
    if config.gpus >= 2:
        model = keras.utils.multi_gpu_model(model, gpus=config.gpus)

    model.compile(loss='binary_crossentropy', 
                optimizer=keras.optimizers.Adam(), 
                metrics=['accuracy'])
    return model

def BuildENModel_Test(embeddings, embedding_dim):
    # --
    # Define the shared model
    x = Sequential()
    x.add(Embedding(len(embeddings), 
                    embedding_dim,
                    weights=[embeddings], 
                    input_shape=(config.en_max_seq_length,), 
                    trainable=False))

    # LSTM
    x.add(Bidirectional(LSTM(config.n_hidden,
                            dropout=config.dropout_rate, 
                            recurrent_dropout=config.dropout_rate,
                            return_sequences=True), name='en_encoder'))
    # x.add(Dropout(0.2))
    x.add(Attention2(config.en_max_seq_length))
    shared_model = x
    
    # The visible layer
    left_input = Input(shape=(config.en_max_seq_length,), dtype='int32')
    right_input = Input(shape=(config.en_max_seq_length,), dtype='int32')

    # Pack it all up into a Manhattan Distance model
    # distance = ManDist()([shared_model(left_input), shared_model(right_input)])
    distance = Euclidean()([shared_model(left_input), shared_model(right_input)])

    model = Model(inputs=[left_input, right_input], 
                outputs=[distance])

    # `multi_gpu_model()` is a so quite buggy. it breaks the saved model.
    if config.gpus >= 2:
        model = keras.utils.multi_gpu_model(model, gpus=config.gpus)

    model.compile(loss='binary_crossentropy', 
                optimizer=keras.optimizers.Adam(), 
                metrics=['accuracy'])
    return model

    
def BuildESModel_baseline(embeddings, embedding_dim):
    # --
    # Define the shared model
    x = Sequential()
    x.add(Embedding(len(embeddings), 
                    embedding_dim,
                    weights=[embeddings], 
                    input_shape=(config.es_max_seq_length,), 
                    trainable=False))

    # LSTM
    x.add(Bidirectional(LSTM(config.n_hidden,
                            # recurrent_dropout=config.dropout_rate,
                            # dropout=config.dropout_rate,
                            return_sequences=True)))
    # x.add(Dropout(0.2))
    x.add(Attention(config.es_max_seq_length))
    shared_model = x
    # The visible layer
    left_input = Input(shape=(config.es_max_seq_length,), dtype='int32')
    right_input = Input(shape=(config.es_max_seq_length,), dtype='int32')

    # Pack it all up into a Manhattan Distance model
    # distance = ManDist()([shared_model(left_input), shared_model(right_input)])
    distance = Euclidean()([shared_model(left_input), shared_model(right_input)])

    model = Model(inputs=[left_input, right_input], 
                outputs=[distance])

    # `multi_gpu_model()` is a so quite buggy. it breaks the saved model.
    if config.gpus >= 2:
        model = keras.utils.multi_gpu_model(model, gpus=config.gpus)

    model.compile(loss='binary_crossentropy', 
                optimizer=keras.optimizers.Adam(), 
                metrics=['accuracy'])
    return model


def LoadTrainData2(fileName):
    df = pd.read_csv(fileName, sep='\t', header=None)
    return df

def text_to_word_list(text):
    # Pre process and convert texts to a list of words
    text = str(text)
    text = text.lower()
    try:
        # Clean the text
        text = re.sub(r'[\"\?)(:;.,\/\'+=-\]\[]',' ', text)
        text = re.sub(r'[0-9]', ' ', text)
    except Exception as e:
        print('#' * 30)
        print(str(e))
        print(text)

    text = text.split()

    return text

def LoadPretrainedEmbeddings(wordFile, vecFile):
    with open(wordFile, 'r', encoding='utf-8') as fword, open(vecFile, 'rb') as fvec:
        info = fword.readline().strip('\n').split('\t')
        vocab_size, embedding_dim = int(info[0]), int(info[1])
        print('vocab_size: ', vocab_size, ' embedding_dim: ', embedding_dim)
        word_to_ix = {}
        for word in fword:
            word = word.strip('\n')
            word_to_ix[word] = len(word_to_ix) + 1 # for the first embedding is zeros
        print('finish load word file')
        embeddings = np.fromfile(fvec, dtype='float', count = vocab_size * embedding_dim).reshape(vocab_size, embedding_dim)
        print('finish load vec file')
        zero_row = np.arange(embedding_dim).reshape(1, embedding_dim)
        print(zero_row.shape)
        print(embeddings.shape)
        embeddings = np.concatenate((zero_row, embeddings), axis=0)
    return word_to_ix, embeddings, embedding_dim
    
def make_DataEmbeddingIndex(df, word_to_ix, embeddings, stops, columns):
    vocabs_not_w2v = set()
    vocabs_not_w2v_cnt = 0
    max_seq_length = -1
    
    # gen word Index
    for index, row in df.iterrows():
        # Print the number of embedded sentences.
        if index != 0 and index % 1000 == 0:
            print("{:,} sentences embedded.".format(index), flush=True)

        # Iterate through the text of both questions of the row
        for sentence in columns:
            q2n = []  # q2n -> sentence numbers representation
            words = text_to_word_list(row[sentence])
            max_seq_length = max(max_seq_length, len(words))
            for word in words:
                # Check for unwanted words
                if word in stops:
                    continue
                    
                # If a word is missing from word2vec model.
                if word not in word_to_ix:
                    if word not in vocabs_not_w2v:
                        vocabs_not_w2v_cnt += 1
                        vocabs_not_w2v.add(word)
                    continue
                else:
                    q2n.append(word_to_ix[word])

            # Append sentence as number representation
            df.at[index, sentence + '_n'] = q2n
    print('vocabs_not_w2v length: ', len(vocabs_not_w2v))
    print('total not w2v: ', vocabs_not_w2v_cnt)
    return df, max_seq_length

def split_and_zero_padding(df, max_seq_length, columns=None):
    data = []
    # Zero padding
    for col in columns:
        data.append(pad_sequences(df[col], padding='pre', truncating='post', maxlen=max_seq_length))
    return data

#  --

class ManDist(Layer):
    """
    Keras Custom Layer that calculates Manhattan Distance.
    """

    # initialize the layer, No need to include inputs parameter!
    def __init__(self, **kwargs):
        self.result = None
        super(ManDist, self).__init__(**kwargs)

    # input_shape will automatic collect input shapes to build layer
    def build(self, input_shape):
        super(ManDist, self).build(input_shape)

    # This is where the layer's logic lives.
    def call(self, x, **kwargs):
        self.result = K.exp(-K.sum(K.abs(x[0] - x[1]), axis=1, keepdims=True))
        return self.result

    # return output shape
    def compute_output_shape(self, input_shape):
        print('heeheheheheheheheeehhhehehe')
        print(self.result.shape)
        print(K.int_shape(self.result))
        return K.int_shape(self.result)

class Euclidean(Layer):
# initialize the layer, No need to include inputs parameter!
    def __init__(self, **kwargs):
        self.result = None
        super(Euclidean, self).__init__(**kwargs)

    # input_shape will automatic collect input shapes to build layer
    def build(self, input_shape):
        super(Euclidean, self).build(input_shape)

    # This is where the layer's logic lives.
    def call(self, x, **kwargs):
        self.result = K.exp(-K.sum((x[0]-x[1])**2, axis=1, keepdims=True))
        return self.result

    # return output shape
    def compute_output_shape(self, input_shape):
        print('heeheheheheheheheeehhhehehe')
        print(self.result.shape)
        print(K.int_shape(self.result))
        return K.int_shape(self.result)


