

class Config(object):
    '''configuration'''
    def __init__(self):
        
        self.es_version = '_0711'
        self.en_version = '_0711'
        # self.version = '_test1'
        # Train data
        self.es_TrainFile = './data/cikm_spanish_train_20180516.txt'
        self.es_embedding_wordFile = './data/wiki.es.vec_word_limit'
        self.es_embedding_vecFile = './data/wiki.es.vec_vec_limit'
        
        self.en_TrainFile = './data/cikm_english_train_20180516.txt'
        self.en_embedding_wordFile = './data/wiki.en.vec_word_limit'
        self.en_embedding_vecFile = './data/wiki.en.vec_vec_limit'

        # Model variables
        self.n_epoch = 1
        self.n_hidden = 50
        self.gpus = 1
        self.batch_size = 128 * self.gpus
        self.lr = 0.001
        self.dropout_rate = 0.2
        self.es_max_seq_length = 60
        self.en_max_seq_length = 60
        self.validation_ratio = 0.2

        # Model save
        self.es_modelPath = './data/es_SiameseLSTM' + self.es_version + '.h5'
        self.es_bst_model_path = './data/es_best' + self.es_version + '.h5'

        self.en_modelPath = './data/en_SiameseLSTM' + self.en_version + '.h5'
        self.en_bst_model_path = './data/en_best' + self.en_version + '.h5'
        
        # Loss plot
        self.es_figurePath = './data/es_history-grap' + self.es_version + '.png'
        self.en_figurePath = './data/en_history-grap' + self.en_version + '.png'
        

        # Prediction
        self.modelForPredict = self.es_bst_model_path
        self.anwserFile = './data/answerFile' + self.es_version + '.txt'
        self.testFile = './data/cikm_test_a_20180516.txt'


        
        