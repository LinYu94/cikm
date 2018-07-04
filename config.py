

class Config(object):
    '''configuration'''
    def __init__(self):
        
        # self.basePath = 'E:/kaggle/CIKM AnalytiCup 2018/CIKM/data'

        self.es_TrainFile = './data/cikm_spanish_train_20180516.txt'
        self.es_embedding_wordFile = './data/wiki.es.vec_word'
        self.es_embedding_vecFile = './data/wiki.es.vec_vec'
        

        self.en_TrainFile = './data/cikm_english_train_20180516.txt'
        self.en_embedding_wordFile = './data/wiki.en.vec_word'
        self.en_embedding_vecFile = './data/wiki.en.vec_vec'

        # Model variables
        self.n_epoch = 50
        self.n_hidden = 50
        self.gpus = 1
        self.batch_size = 256 * self.gpus

        self.modelPath = './data/SiameseLSTM_0628.h5'
        self.bst_model_path = './data/best_0704.h5'
        self.modelForPredict = self.bst_model_path
        self.figurePath = './data/history-grap-0628.png'

        self.max_seq_length = 60
        self.validation_ratio = 0.3

        # Test
        self.AnwserFile = './data/answerFile_0628.txt'
        self.TestFile = './data/cikm_test_a_20180516.txt'

        self.lr = 0.001
        self.dropout_rate = 0.1
        
        