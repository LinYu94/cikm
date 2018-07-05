import re

from keras import backend as K
from keras.layers import Layer
from keras.preprocessing.sequence import pad_sequences

from nltk.corpus import stopwords
from gensim.models import KeyedVectors
import tensorflow as tf
from config import Config
import gensim

import numpy as np

import itertools

import pandas as pd

def BuildESModel():
    # --
    # Define the shared model
    x = Sequential()
    x.add(Embedding(len(embeddings), embedding_dim,
                    weights=[embeddings], input_shape=(config.max_seq_length,), trainable=False))
    # CNN
    # x.add(Conv1D(250, kernel_size=5, activation='relu'))
    # x.add(GlobalMaxPool1D())
    # x.add(Dense(250, activation='relu'))
    # x.add(Dropout(0.3))
    # x.add(Dense(1, activation='sigmoid'))
    # LSTM
    x.add(Bidirectional(LSTM(config.n_hidden,
                            dropout=config.dropout_rate, 
                            recurrent_dropout=config.dropout_rate)))

    # x.add(Bidirectional(LSTM(config.n_hidden, 
    #                         dropout=config.dropout_rate, 
    #                         recurrent_dropout=config.dropout_rate)))

    shared_model = x

    # The visible layer
    left_input = Input(shape=(config.max_seq_length,), dtype='int32')
    right_input = Input(shape=(config.max_seq_length,), dtype='int32')

    # Pack it all up into a Manhattan Distance model
    # malstm_distance = ManDist()([shared_model(left_input), shared_model(right_input)])

    concatenated = keras.layers.concatenate([shared_model(left_input), shared_model(right_input)])
    out = Dense(1, activation='sigmoid')(concatenated)

    # model = Model(inputs=[left_input, right_input], outputs=[malstm_distance])
    model = Model(inputs=[left_input, right_input], outputs=out)

    # `multi_gpu_model()` is a so quite buggy. it breaks the saved model.
    if config.gpus >= 2:
        model = keras.utils.multi_gpu_model(model, gpus=config.gpus)

    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    model_checkpoint = ModelCheckpoint(config.bst_model_path, save_best_only=True)

    model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
    model.summary()
    shared_model.summary()


def BuildENModel(embeddings, embedding_dim):
    # --
    # Define the shared model
    x = Sequential()
    x.add(Embedding(len(embeddings), embedding_dim,
                    weights=[embeddings], input_shape=(config.en_max_seq_length,), trainable=False))

    # LSTM
    x.add(Bidirectional(LSTM(config.n_hidden,
                            dropout=config.dropout_rate, 
                            recurrent_dropout=config.dropout_rate)))

    shared_model = x
    # The visible layer
    left_input = Input(shape=(config.en_max_seq_length,), dtype='int32')
    right_input = Input(shape=(config.en_max_seq_length,), dtype='int32')

    # Pack it all up into a Manhattan Distance model
    malstm_distance = ManDist()([shared_model(left_input), shared_model(right_input)])

    model = Model(inputs=[left_input, right_input], outputs=[malstm_distance])

    # `multi_gpu_model()` is a so quite buggy. it breaks the saved model.
    if config.gpus >= 2:
        model = keras.utils.multi_gpu_model(model, gpus=config.gpus)



    model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
    model.summary()
    shared_model.summary()



def LoadTrainData(fileName):
    df = pd.read_csv(fileName, sep='\t', header=None)
    df.columns = ['es1', 'en1', 'es2', 'en2', 'sim']
    data = [[],[]]
    for i in range(len(df['sim'])):
        data[0].append((df.iloc[i, 0], df.iloc[i, 2]))
        data[1].append(df.iloc[i, 4])
    return data

def LoadTrainData2(fileName):
    df = pd.read_csv(fileName, sep='\t', header=None)
    return df

def LoadTestData(fileName):
    df = pd.read_csv(fileName, sep='\t', header=None)
    df.columns = ['es1', 'es2']
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



def split_and_zero_padding(df, max_seq_length):
    # Zero padding
    df['es1_n'] = pad_sequences(df['es1_n'], padding='pre', truncating='post', maxlen=max_seq_length)
    df['es2_n'] = pad_sequences(df['es2_n'], padding='pre', truncating='post', maxlen=max_seq_length)
    return df

def split_and_zero_padding_en(df, max_seq_length):
    # Zero padding
    df['en1_n'] = pad_sequences(df['en1_n'], padding='pre', truncating='post', maxlen=max_seq_length)
    df['en2_n'] = pad_sequences(df['en2_n'], padding='pre', truncating='post', maxlen=max_seq_length)
    return df

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


class EmptyWord2Vec:
    """
    Just for test use.
    """
    vocab = {}
    word_vec = {}
