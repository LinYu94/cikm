import re

from keras import backend as K
from keras.layers import Layer
from keras.preprocessing.sequence import pad_sequences

from nltk.corpus import stopwords
from gensim.models import KeyedVectors
import tensorflow as tf
from config_dev import Config
import gensim
import keras
from keras.models import Model, Sequential
from keras.layers import Input, Embedding, LSTM, GRU, Conv1D, Conv2D, GlobalMaxPool1D, Dense, Dropout, Bidirectional, Lambda
from keras.layers import BatchNormalization

import numpy as np
import itertools
import pandas as pd

config = Config()

def BuildESModel3(es_embeddings, es_embedding_dim, en_embeddings, en_embedding_dim):

    # Define the shared model
    es_encoder = Sequential()
    es_encoder.add(Embedding(len(es_embeddings), es_embedding_dim,
                    weights=[es_embeddings], 
                    input_shape=(config.es_max_seq_length,),
                    trainable=False))
    
    # LSTM
    es_encoder.add(Bidirectional(LSTM(config.n_hidden,
                            dropout=config.dropout_rate, 
                            recurrent_dropout=config.dropout_rate), name='es_encoder'))

    en_encoder = Sequential()
    en_encoder.add(Embedding(len(en_embeddings), en_embedding_dim,
                    weights=[en_embeddings], 
                    input_shape=(config.en_max_seq_length,),
                    trainable=False))
    
    # LSTM
    en_encoder.add(Bidirectional(LSTM(config.n_hidden,
                            dropout=config.dropout_rate, 
                            recurrent_dropout=config.dropout_rate), name='en_encoder'))


    # The visible layer
    en1_input = Input(shape=(config.en_max_seq_length,), dtype='int32')
    en2_input = Input(shape=(config.en_max_seq_length,), dtype='int32')

    es1_input = Input(shape=(config.es_max_seq_length,), dtype='int32')
    es2_input = Input(shape=(config.es_max_seq_length,), dtype='int32')

    # Pack it all up into a distance methord
    if config.distance == "ManDist":
        ss_malstm_distance = Lambda(ManDist)([es_encoder(es1_input), es_encoder(es2_input)])
        nn_malstm_distance = Lambda(ManDist)([en_encoder(en1_input), en_encoder(en2_input)])
        s1n1_malstm_distance = Lambda(ManDist)([es_encoder(es1_input), en_encoder(en1_input)])
        s2n2_malstm_distance = Lambda(ManDist)([es_encoder(es2_input), en_encoder(en2_input)])
    elif config.distance == "Euclidean":
        ss_malstm_distance = Lambda(Euclidean)([es_encoder(es1_input), es_encoder(es2_input)])
        nn_malstm_distance = Lambda(Euclidean)([en_encoder(en1_input), en_encoder(en2_input)])
        s1n1_malstm_distance = Lambda(Euclidean)([es_encoder(es1_input), en_encoder(en1_input)])
        s2n2_malstm_distance = Lambda(Euclidean)([es_encoder(es2_input), en_encoder(en2_input)])
    elif config.distance == "Cosine":
        ss_malstm_distance = Lambda(Cosine)([es_encoder(es1_input), es_encoder(es2_input)])
        nn_malstm_distance = Lambda(Cosine)([en_encoder(en1_input), en_encoder(en2_input)])
        s1n1_malstm_distance = Lambda(Cosine)([es_encoder(es1_input), en_encoder(en1_input)])
        s2n2_malstm_distance = Lambda(Cosine)([es_encoder(es2_input), en_encoder(en2_input)])

    model = Model(inputs=[en1_input, en2_input, es1_input, es2_input], 
                outputs=[ss_malstm_distance, nn_malstm_distance, s1n1_malstm_distance, s2n2_malstm_distance])

    model.compile(loss='binary_crossentropy', 
                  optimizer=keras.optimizers.Adam(),
                  loss_weights=[1, 0.9, 0.9, 0.9],
                  metrics=['accuracy'])
    return model


def BuildESModel(embeddings, embedding_dim):
    # --
    # Load EN model
    print('Loading EN encoder...: ', config.en_bst_model_path)
    enModel = keras.models.load_model(config.en_bst_model_path, custom_objects={'ManDist': ManDist})
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
                  loss_weights=[1, 0.2, 0.2],
                  metrics=['accuracy'])
    print(model.summary())
    shared_model.summary()
    return model

def BuildESModel2(embeddings, embedding_dim):
    # --
    # Load EN model
    print('Loading EN encoder...: ', config.en_bst_model_path)
    enModel = keras.models.load_model(config.en_bst_model_path, custom_objects={'ManDist': ManDist})
    enEncoder = enModel.layers[2]
    print(enEncoder)
    # for layer in enEncoder.layers:
    #     layer.trainable = False
    
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
    en_left_input = Input(shape=(config.en_max_seq_length,), dtype='int32')
    en_right_input = Input(shape=(config.en_max_seq_length,), dtype='int32')

    es_left_input = Input(shape=(config.es_max_seq_length,), dtype='int32')
    es_right_input = Input(shape=(config.es_max_seq_length,), dtype='int32')

    # Pack it all up into a Manhattan Distance model
    # malstm_distance = ManDist()([shared_model(left_input), shared_model(right_input)])

    ss_malstm_distance = ManDist()([shared_model(es_left_input), shared_model(es_right_input)])
    s1n1_malstm_distance = ManDist()([shared_model(es_left_input), enEncoder(en_left_input)])
    s2n2_malstm_distance = ManDist()([shared_model(es_right_input), enEncoder(en_right_input)])

    model = Model(inputs=[es_left_input, es_right_input, en_left_input, en_right_input], 
                outputs=[ss_malstm_distance, s1n1_malstm_distance, s2n2_malstm_distance])

    # `multi_gpu_model()` is a so quite buggy. it breaks the saved model.
    if config.gpus >= 2:
        model = keras.utils.multi_gpu_model(model, gpus=config.gpus)

    model.compile(loss='binary_crossentropy', 
                  optimizer=keras.optimizers.Adam(),
                  loss_weights=[1, 1, 1],
                  metrics=['accuracy'])
    model.summary()
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
                            dropout=config.dropout_rate, 
                            recurrent_dropout=config.dropout_rate), name='en_encoder'))

    shared_model = x
    # The visible layer
    left_input = Input(shape=(config.en_max_seq_length,), dtype='int32')
    right_input = Input(shape=(config.en_max_seq_length,), dtype='int32')

    # Pack it all up into a Manhattan Distance model
    # distance = ManDist()([shared_model(left_input), shared_model(right_input)])
    distance = ManDist()([shared_model(left_input), shared_model(right_input)])

    model = Model(inputs=[left_input, right_input], 
                outputs=[distance])

    # `multi_gpu_model()` is a so quite buggy. it breaks the saved model.
    if config.gpus >= 2:
        model = keras.utils.multi_gpu_model(model, gpus=config.gpus)

    model.compile(loss='binary_crossentropy', 
                optimizer=keras.optimizers.Adam(), 
                metrics=['accuracy'])

    print(model.summary())
    shared_model.summary()
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
    # def compute_output_shape(self, input_shape):
    #     print('input_shape: ', input_shape)
    #     return (len(input_shape), self.output_dim)

    # return output shape
    def compute_output_shape(self, input_shape):
        return K.int_shape(self.result)

def ManDist(x):
    return K.exp(-K.sum(K.abs(x[0] - x[1]), axis=1, keepdims=True))

def Euclidean(x):
    return K.exp(-K.sum((x[0]-x[1])**2, axis=1, keepdims=True))


def Cosine(x):
    norm_x0 = x[0] / K.sqrt(K.sum(x[0]**2, axis=1, keepdims=True))
    norm_x1 = x[1] / K.sqrt(K.sum(x[1]**2, axis=1, keepdims=True))
    return (1 + K.dot(norm_x0, K.transpose(norm_x1))) / 2