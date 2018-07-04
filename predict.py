import pandas as pd

import tensorflow as tf
import keras
from util import LoadTestData, LoadPretrainedEmbeddings, make_DataEmbeddingIndex
from util import split_and_zero_padding
from util import ManDist
import numpy as np
from config import Config
import gc

config = Config()
# Load training set
test_df = LoadTestData(config.TestFile)
for q in ['es1', 'es2']:
    test_df[q + '_n'] = test_df[q]


# Load Pretrained WordEmbeddings
word_to_ix, embeddings, embedding_dim = LoadPretrainedEmbeddings(config.es_embedding_wordFile, config.es_embedding_vecFile)

# Make testData embeddings index
test_df, max_seq_length = make_DataEmbeddingIndex(test_df, word_to_ix, embeddings)

print('max_seq_length: ', max_seq_length)

del embeddings
# Split to dicts and append zero padding.
X_test = split_and_zero_padding(test_df, config.max_seq_length)

# Make sure everything is ok
assert X_test['left'].shape == X_test['right'].shape

# --

# model = keras.models.load_model(config.modelPath, custom_objects={'ManDist': ManDist})
model = keras.models.load_model(config.modelForPredict)
model.summary()
print(type(X_test['left']))
print(X_test['left'].shape)


prediction = model.predict([X_test['left'], X_test['right']])
with open(config.AnwserFile, 'w') as fw:
    for x in prediction:
        fw.write(str(x[0]) + '\n')
print('predict finish!')
