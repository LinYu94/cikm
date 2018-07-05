

class Config(object):
    '''configuration'''
    def __init__(self):
        
        self.version = '_0705'

        # Train data
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
        self.lr = 0.001
        self.dropout_rate = 0.1
        self.es_max_seq_length = 60
        self.en_max_seq_length = 60
        self.validation_ratio = 0.3

        # Model save
        self.es_modelPath = './data/SiameseLSTM' + self.version + '.h5'
        self.es_bst_model_path = './data/es_best' + self.version + '.h5'

        self.en_modelPath = './data/SiameseLSTM' + self.version + '.h5'
        self.en_bst_model_path = './data/en_best' + self.version + '.h5'
        
        # Loss plot
        self.es_figurePath = './data/es_history-grap' + self.version + '.png'
        self.en_figurePath = './data/en_history-grap' + self.version + '.png'
        

        # Prediction
        self.modelForPredict = self.bst_model_path
        self.anwserFile = './data/answerFile' + self.version + '.txt'
        self.testFile = './data/cikm_test_a_20180516.txt'


        
        