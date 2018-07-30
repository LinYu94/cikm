from time import time
import pandas as pd

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import keras
import tensorflow as tf
from keras.models import Model, Sequential
from keras.layers import Input, Embedding, LSTM, GRU, Conv1D, Conv2D, GlobalMaxPool1D, Dense, Dropout, Bidirectional
from keras.layers import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint

from util import *
from config_dev import Config

import gc
from nltk.corpus import stopwords

config = Config()
# Load Pretrained WordEmbeddings
es_word_to_ix, es_embeddings, es_embedding_dim = LoadPretrainedEmbeddings(config.es_embedding_wordFile, config.es_embedding_vecFile)
en_word_to_ix, en_embeddings, en_embedding_dim = LoadPretrainedEmbeddings(config.en_embedding_wordFile, config.en_embedding_vecFile)

# Load training set1
train_df1 = LoadTrainData2(config.es_TrainFile)
train_df1.columns = ['es1', 'en1', 'es2', 'en2', 'sim']
for q in ['es1', 'en1', 'es2', 'en2']:
    train_df1[q + '_n'] = train_df1[q]

# Load training set2(en traslation)
train_df2 = LoadTrainData2(config.en_TrainFile)
train_df2.columns = ['en1', 'es1', 'en2', 'es2', 'sim']
for q in ['en1', 'es1', 'en2', 'es2']:
    train_df2[q + '_n'] = train_df2[q]


en_stops = set(stopwords.words('english'))
es_stops = set(stopwords.words('spanish'))
# Make trainData embeddings index
train_df1, max_seq_length = make_DataEmbeddingIndex(train_df1, es_word_to_ix, es_embeddings, es_stops, columns=['es1', 'es2'])
print('train data1 max_seq_length: ', max_seq_length)

train_df1, max_seq_length = make_DataEmbeddingIndex(train_df1, en_word_to_ix, en_embeddings, en_stops, columns=['en1', 'en2'])
print('train data1 max_seq_length: ', max_seq_length)

train_df2, max_seq_length = make_DataEmbeddingIndex(train_df2, es_word_to_ix, es_embeddings, es_stops, columns=['es1', 'es2'])
print('train data2 max_seq_length: ', max_seq_length)

train_df2, max_seq_length = make_DataEmbeddingIndex(train_df2, en_word_to_ix, en_embeddings, en_stops, columns=['en1', 'en2'])
print('train data2 max_seq_length: ', max_seq_length)

train_df1 = train_df1[['es1_n', 'es2_n', 'en1_n', 'en2_n', 'sim']]
train_df2 = train_df2[['es1_n', 'es2_n', 'en1_n', 'en2_n', 'sim']]


train_df = pd.concat((train_df1, train_df2))

# shuffle train data 
train_df = shuffle(train_df)

# Split to train validation
validation_size = int(len(train_df) * config.validation_ratio)
training_size = len(train_df) - validation_size


validation_df = train_df[:validation_size]
train_df = train_df[validation_size:]


X_train = train_df[['es1_n', 'es2_n', 'en1_n', 'en2_n']]
Y_train = train_df['sim']
X_validation = validation_df[['es1_n', 'es2_n', 'en1_n', 'en2_n']]
Y_validation = validation_df['sim']


# es_max_seq_length == en_max_seq_length == 60
X_train = split_and_zero_padding(X_train, config.es_max_seq_length, columns=['es1_n', 'es2_n', 'en1_n', 'en2_n'])
X_validation = split_and_zero_padding(X_validation, config.es_max_seq_length, columns=['es1_n', 'es2_n', 'en1_n', 'en2_n'])


# Save best, Early Stop
model_checkpoint = ModelCheckpoint(config.es_bst_model_path, monitor='val_loss', save_best_only=True, save_weights_only=True)



# Start trainings
training_start_time = time()
model = BuildESModel2(es_embeddings, es_embedding_dim, en_embeddings, en_embedding_dim)
model.summary()
malstm_trained = model.fit(X_train, [Y_train, Y_train, Y_train, Y_train],
                           batch_size=config.batch_size, 
                           epochs=config.n_epoch, 
                           shuffle=True,
                           verbose=2,
                           callbacks=[model_checkpoint],
                           validation_data=(X_validation, [Y_validation, Y_validation, Y_validation, Y_validation]))
training_end_time = time()
print("Training time finished.\n%d epochs in %12.2f" % (config.n_epoch,
                                                        training_end_time - training_start_time))

# model.save(config.es_modelPath)

# Plot accuracy
# plt.subplot(211)
# plt.plot(malstm_trained.history['acc'])
# plt.plot(malstm_trained.history['val_acc'])
# plt.title('Model Accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Validation'], loc='upper left')

# Plot loss
# plt.subplot(211)
# plt.plot(malstm_trained.history['loss'])
# plt.plot(malstm_trained.history['val_loss'])
# plt.title('Model Loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Validation'], loc='upper right')

# plt.tight_layout(h_pad=1.0)
# plt.savefig(config.es_figurePath)

# print(str(malstm_trained.history['val_acc'][-1])[:6] +
#       "(max: " + str(max(malstm_trained.history['val_acc']))[:6] + ")")
print("Done.")
