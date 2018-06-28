import pandas as pd

import tensorflow as tf
import keras
from util import LoadTestData, LoadPretrainedEmbeddings, make_DataEmbeddingIndex
from util import split_and_zero_padding
from util import ManDist
import numpy as np
import gc
# File paths
TEST_FILE = './data/cikm_test_a_20180516.txt'

# Load training set
test_df = LoadTestData(TEST_FILE)
for q in ['es1', 'es2']:
    test_df[q + '_n'] = test_df[q]

EMBEDDING_WORD = './data/wiki.es.vec_word'
EMBEDDING_VEC = './data/wiki.es.vec_vec'

# Load Pretrained WordEmbeddings
word_to_ix, embeddings, embedding_dim = LoadPretrainedEmbeddings(EMBEDDING_WORD, EMBEDDING_VEC)

# Make testData embeddings index
test_df, max_seq_length = make_DataEmbeddingIndex(test_df, word_to_ix, embeddings)

print('max_seq_length: ', max_seq_length)
max_seq_length = 60

del embeddings
# Split to dicts and append zero padding.
X_test = split_and_zero_padding(test_df, max_seq_length)

# Make sure everything is ok
assert X_test['left'].shape == X_test['right'].shape

# --

model = keras.models.load_model('./data/SiameseLSTM_0628.h5', custom_objects={'ManDist': ManDist})
model.summary()
print(type(X_test['left']))
print(X_test['left'].shape)
# for i in range(len(X_test['left'])):
#     print(X_test['left'][i], '\t###\t', X_test['right'][i])
#     try:
#         prediction = model.predict([np.array(X_test['left'][i])], [np.array(X_test['right'][i])])
#         print(prediction)
#     except Exception as e:
#         print('#' * 100)
#         print('wrong case: ', i)
#         print(str(e))
#         break


prediction = model.predict([X_test['left'], X_test['right']])
print(prediction)
with open('./data/answer_0628.txt', 'w') as fw:
    for x in prediction:
        fw.write(str(x[0]) + '\n')
print('predict finish!')
