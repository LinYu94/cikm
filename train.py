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

from util import LoadTrainData2, LoadPretrainedEmbeddings, make_DataEmbeddingIndex
from util import split_and_zero_padding
from util import ManDist
from config import Config

import gc

config = Config()

# Load Pretrained WordEmbeddings
word_to_ix, embeddings, embedding_dim = LoadPretrainedEmbeddings(config.es_embedding_wordFile, config.es_embedding_vecFile)

# Load training set1
train_df1 = LoadTrainData2(config.es_TrainFile)
train_df1.columns = ['es1', 'en1', 'es2', 'en2', 'sim']
for q in ['es1', 'es2']:
    train_df1[q + '_n'] = train_df1[q]

# Load training set2(en traslation)
train_df2 = LoadTrainData2(config.en_TrainFile)
train_df2.columns = ['en1', 'es1', 'en2', 'es2', 'sim']
for q in ['es1', 'es2']:
    train_df2[q + '_n'] = train_df2[q]




# Make trainData embeddings index
train_df1, max_seq_length = make_DataEmbeddingIndex(train_df1, word_to_ix, embeddings)
print('train data1 max_seq_length: ', max_seq_length)

train_df2, max_seq_length = make_DataEmbeddingIndex(train_df2, word_to_ix, embeddings)
print('train data2 max_seq_length: ', max_seq_length)


train_df1 = train_df1[['es1_n', 'es2_n', 'sim']]
train_df2 = train_df2[['es1_n', 'es2_n', 'sim']]

# gc.collect()
train_df = pd.concat((train_df1, train_df2))

# after construct train df, garbage clean
print('embedding_dim: ', embedding_dim)
print('vocab size +1: ', len(embeddings))


# shuffle train data 
train_df = shuffle(train_df)

# Split to train validation
validation_size = int(len(train_df) * config.validation_ratio)
training_size = len(train_df) - validation_size


X = train_df[['es1_n', 'es2_n']]
Y = train_df['sim']

X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size)

X_train = split_and_zero_padding(X_train, config.max_seq_length)
X_validation = split_and_zero_padding(X_validation, config.max_seq_length)

# Convert labels to their numpy representations
Y_train = Y_train.values
Y_validation = Y_validation.values

# Make sure everything is ok
assert X_train['left'].shape == X_train['right'].shape
assert len(X_train['left']) == len(Y_train)

# --
DSSM_DIM = 25
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

x.add(Dense(DSSM_DIM, activation='relu'))
x.add(BatchNormalization())

x.add(Dense(DSSM_DIM, activation='tanh'))
x.add(BatchNormalization())

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
    model = tf.keras.utils.multi_gpu_model(model, gpus=config.gpus)

early_stopping = EarlyStopping(monitor='val_loss', patience=5)
model_checkpoint = ModelCheckpoint(config.bst_model_path, save_best_only=True)

model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
model.summary()
shared_model.summary()

# Start trainings
training_start_time = time()
malstm_trained = model.fit([X_train['left'], X_train['right']], Y_train,
                           batch_size=config.batch_size, epochs=config.n_epoch, shuffle=True,
                           validation_data=([X_validation['left'], X_validation['right']], Y_validation),
                           callbacks=[early_stopping, model_checkpoint])
training_end_time = time()
print("Training time finished.\n%d epochs in %12.2f" % (config.n_epoch,
                                                        training_end_time - training_start_time))

model.save(config.modelPath)

# Plot accuracy
plt.subplot(211)
plt.plot(malstm_trained.history['acc'])
plt.plot(malstm_trained.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot loss
plt.subplot(212)
plt.plot(malstm_trained.history['loss'])
plt.plot(malstm_trained.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')

plt.tight_layout(h_pad=1.0)
plt.savefig(config.figurePath)

print(str(malstm_trained.history['val_acc'][-1])[:6] +
      "(max: " + str(max(malstm_trained.history['val_acc']))[:6] + ")")
print("Done.")
