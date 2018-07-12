import pandas as pd

import tensorflow as tf
import keras
from util import *
import numpy as np
from config import Config
import gc

config = Config()
# Load training set
test_df = LoadTrainData2(config.testFile)
test_df.columns = ['es1', 'es2']
for q in ['es1', 'es2']:
    test_df[q + '_n'] = test_df[q]


# Load Pretrained WordEmbeddings
word_to_ix, embeddings, embedding_dim = LoadPretrainedEmbeddings(config.es_embedding_wordFile, config.es_embedding_vecFile)


stops = set(stopwords.words('spanish'))
# Make testData embeddings index
test_df, max_seq_length = make_DataEmbeddingIndex(test_df, word_to_ix, embeddings, stops, columns=['es1', 'es2'])
print('test data max_seq_length: ', max_seq_length)

# del embeddings
# Split to dicts and append zero padding.
X_test = split_and_zero_padding(test_df, config.es_max_seq_length, columns=['es1_n', 'es2_n'])

print(type(X_test))
print(len(X_test))
print(len(X_test[0]))
# --

# model = keras.models.load_model(config.modelForPredict, custom_objects={'Euclidean': Euclidean})
# model.summary()
model = BuildENModel(embeddings, embedding_dim)
model.load_weights(config.es_bst_model_path)
model.summary()


# prediction = model.predict([X_test[0], X_test[1], X_test[0], X_test[0]])
prediction = model.predict([X_test[0], X_test[1]])
print(type(prediction))
print(prediction)
with open(config.anwserFile, 'w') as fw:
    for x in prediction:
        fw.write(str(x[0]) + '\n')
# with open('all', 'w') as fw:
#     for x in prediction:
#         fw.write('\t'.join(x) + '\n')
print('predict finish!')
