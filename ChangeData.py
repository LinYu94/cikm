
import numpy as np
from array import array
import sys

class ConvertWordEmbeddingFile():
    def __init__(self):
        pass
    def __call__(self, filePath):
        with open(filePath, 'r', encoding='utf-8') as f, open(filePath + '_word', 'w', encoding='utf-8') as fword,\
                open(filePath + '_vec', 'wb') as fvec:
            arr = f.readline().split(' ')
            word_num, embedding_dim = int(arr[0]), int(arr[1])
            print(filePath, ': ', word_num, '\t', embedding_dim)
            fword.write(str(word_num) + '\t' + str(embedding_dim) + '\n')
            for i, line in enumerate(f):
                if i % 100000 == 0:
                    print(i)
                arr = line.split(' ')
                word = arr[0]
                fword.write(word + '\n')

                emb = [float(x) for x in arr[1: embedding_dim + 1]]
                float_array = array('d', emb)
                float_array.tofile(fvec)
            print(filePath, ' finish!')

if __name__ == '__main__':
    cwef = ConvertWordEmbeddingFile()
    for filePath in sys.argv[1:]:
        cwef(filePath)
