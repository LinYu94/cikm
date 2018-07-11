from time import time
import pandas as pd

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import keras
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint
from util import *
from config import Config

import gc
from nltk.corpus import stopwords
config = Config()



# Load Pretrained WordEmbeddings
word_to_ix, embeddings, embedding_dim = LoadPretrainedEmbeddings(config.en_embedding_wordFile, config.en_embedding_vecFile)

# Load training set1
train_df1 = LoadTrainData2(config.es_TrainFile)
train_df1.columns = ['es1', 'en1', 'es2', 'en2', 'sim']
for q in ['en1', 'en2']:
    train_df1[q + '_n'] = train_df1[q]

# Load training set2(en traslation)
train_df2 = LoadTrainData2(config.en_TrainFile)
train_df2.columns = ['en1', 'es1', 'en2', 'es2', 'sim']
for q in ['en1', 'en2']:
    train_df2[q + '_n'] = train_df2[q]

# Stopwords
stops = set(stopwords.words('english'))

# Make trainData embeddings index
train_df1, max_seq_length = make_DataEmbeddingIndex(train_df1, word_to_ix, embeddings, stops, columns=['en1', 'en2'])
print('train data1 max_seq_length: ', max_seq_length)

train_df2, max_seq_length = make_DataEmbeddingIndex(train_df2, word_to_ix, embeddings, stops, columns=['en1', 'en2'] )
print('train data2 max_seq_length: ', max_seq_length)


train_df1 = train_df1[['en1_n', 'en2_n', 'sim']]
train_df2 = train_df2[['en1_n', 'en2_n', 'sim']]

# gc.collect()
train_df = pd.concat((train_df1, train_df2))

# after construct train df, garbage clean
print('embedding_dim: ', embedding_dim)
print('vocab size +1: ', len(embeddings))
print('train data size: ', len(train_df))




# shuffle train data 
train_df = shuffle(train_df)

# Split to train validation
validation_size = int(len(train_df) * config.validation_ratio)
training_size = len(train_df) - validation_size

X = train_df[['en1_n', 'en2_n']]
Y = train_df['sim']
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size)

# Padding Zero
X_train = split_and_zero_padding(X_train, config.en_max_seq_length, ['en1_n', 'en2_n'])
X_validation = split_and_zero_padding(X_validation, config.en_max_seq_length, ['en1_n', 'en2_n'])



# Save best, Early Stop
early_stopping = EarlyStopping(monitor='val_loss', patience=5)
model_checkpoint = ModelCheckpoint(config.en_bst_model_path, monitor='val_loss', save_best_only=True)

model = BuildENModel(embeddings, embedding_dim)
# Start trainings
training_start_time = time()
malstm_trained = model.fit(X_train, Y_train,
                           batch_size=config.batch_size, 
                           epochs=config.n_epoch, 
                           shuffle=True,
                           verbose=2,
                        #    callbacks=[model_checkpoint],
                           validation_data=(X_validation, Y_validation)) # here do not early stop
training_end_time = time()
print("Training time finished.\n%d epochs in %12.2fs" % (config.n_epoch,
                                                        training_end_time - training_start_time))
model.save(config.en_bst_model_path)

# # Plot accuracy
# plt.subplot(211)
# plt.plot(malstm_trained.history['acc'])
# plt.plot(malstm_trained.history['val_acc'])
# plt.title('Model Accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Validation'], loc='upper left')

# # Plot loss
# plt.subplot(212)
# plt.plot(malstm_trained.history['loss'])
# plt.plot(malstm_trained.history['val_loss'])
# plt.title('Model Loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Validation'], loc='upper right')

# plt.tight_layout(h_pad=1.0)
# plt.savefig(config.en_figurePath)

# print(str(malstm_trained.history['val_acc'][-1])[:6] +
#       "(max: " + str(max(malstm_trained.history['val_acc']))[:6] + ")")
# print("Done.")
