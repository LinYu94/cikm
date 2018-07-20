import pandas as pd

import tensorflow as tf
import keras
from util import *
import numpy as np
from config_dev import Config
import gc

config = Config()
# Load training set
test_df = LoadTrainData2(config.testFile)
test_df.columns = ['es1', 'es2']
for q in ['es1', 'es2']:
    test_df[q + '_n'] = test_df[q]


# Load Pretrained WordEmbeddings
es_word_to_ix, es_embeddings, es_embedding_dim = LoadPretrainedEmbeddings(config.es_embedding_wordFile, config.es_embedding_vecFile)
en_word_to_ix, en_embeddings, en_embedding_dim = LoadPretrainedEmbeddings(config.en_embedding_wordFile, config.en_embedding_vecFile)

stops = set(stopwords.words('spanish'))
# Make testData embeddings index
test_df, max_seq_length = make_DataEmbeddingIndex(test_df, es_word_to_ix, es_embeddings, stops, columns=['es1', 'es2'])
print('test data max_seq_length: ', max_seq_length)


# Split to dicts and append zero padding.
X_test = split_and_zero_padding(test_df, config.es_max_seq_length, columns=['es1_n', 'es2_n'])

print(type(X_test))
print(len(X_test))
print(len(X_test[0]))
# --

model = BuildESModel3(es_embeddings, es_embedding_dim, en_embeddings, en_embedding_dim)
model.load_weights(config.es_bst_model_path)


prediction = model.predict([X_test[0], X_test[1], X_test[0], X_test[0]])

print(type(prediction))
print(prediction)
with open(config.anwserFile, 'w') as fw:
    for x in prediction[0]:
        fw.write(str(x[0]) + '\n')
with open('all', 'w') as fw:
    for x in prediction:
        fw.write('\t'.join(x) + '\n')
print('predict finish!')
