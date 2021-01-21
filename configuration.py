import os


def mkdir_if_not_exists(path):
    if not os.path.isdir(path):
        os.mkdir(path)
        return True
    return False


class ConfigPaths(object):
    def __init__(self, configurations, rootdir='./data/'):
        self._root = rootdir if rootdir[-1] == '/' else rootdir + '/'
        self._texts = 'texts/'
        self._batches = 'batches/'
        self._vectors = 'vectors/'
        self._embeddings = 'embeddings/'
        self._doc_data_text = 'extract_data.txt'
        self._doc_labels_text = 'extract_labels.txt'
        self._doc_data_shuffled = 'shuffled_data.txt'
        self._doc_labels_shuffled = 'shuffled_labels.txt'
        self._batch_info = 'batch_info'
        self._vocab_data = 'idx2word_data'
        self._vocab_labels = 'idx2word_labels'
        self._matrices_data = 'matrices_data'
        self._matrices_labels = 'matrices_labels'

        self._config = None
        self._config_vec = None

        if 'wiki-sk-title-intro' in configurations:
            self._config = 'wiki-sk-title-intro/'

        if 'wiki-en-intro-fulltext' in configurations:
            self._config = 'wiki-en-intro-fulltext/'

        if 'annota-title-abstract' in configurations:
            self._config = 'annota-title-abstract/'

        if 'annota-title-fulltext' in configurations:
            self._config = 'annota-title-fulltext/'

        if 'vectors/sk/80' in configurations:
            self._config_vec = 'sk/80/'
            self._file = 'prim-6.1-public-all.shuffled.80cbow.bin'

        if 'vectors/sk/200' in configurations:
            self._config_vec = 'sk/200/'
            self._file = 'prim-6.1-public-all.shuffled.200cbow.bin'

        if 'vectors/en/300' in configurations:
            self._config_vec = 'en/300/'
            self._file = 'GoogleNews-vectors-negative300.bin.gz.bin'

        if self._config:
            self.doc_data_text = self._root + self._texts + self._config + self._doc_data_text
            self.doc_labels_text = self._root + self._texts + self._config + self._doc_labels_text

            self.doc_data_shuffled = self._root + self._texts + self._config + self._doc_data_shuffled
            self.doc_labels_shuffled = self._root + self._texts + self._config + self._doc_labels_shuffled

            self.batch_info = self._root + self._batches + self._config + self._batch_info

            self.vocab_data = self._root + self._batches + self._config + self._vocab_data
            self.vocab_labels = self._root + self._batches + self._config + self._vocab_labels

        if self._config_vec:
            self.vectors = self._root + self._vectors + self._config_vec + self._file
            self.embeddings_data = self._root + self._embeddings + self._config + self._config_vec + \
                                   self._matrices_data
            self.embeddings_labels = self._root + self._embeddings + self._config + self._config_vec + \
                                     self._matrices_labels

    def batch_data(self, type, num):
        mkdir_if_not_exists(self._root + self._batches + self._config + type)
        return self._root + self._batches + self._config + type + '/batch_data_{}'.format(num)

    def batch_labels(self, type, num):
        mkdir_if_not_exists(self._root + self._batches + type)
        return self._root + self._batches + self._config + type + '/batch_labels_{}'.format(num)


class ConfParams(object):
    def __init__(self, configurations):
        self.doc_random_seed = 1653414405048039020

        self.lowercase = False
        self.batch_size = 32
        self.embedding_dim = 100
        self.attention = False
        if 'wiki-sk-title-intro' in configurations:
            self.doc_data_max_words = 50
            self.doc_labels_max_words = 5
        if 'wiki-en-intro-fulltext' in configurations:
            self.doc_data_max_words = 150
            self.doc_labels_max_words = 30
        if 'annota-title-abstract' in configurations:
            self.doc_data_max_words = 150
            self.doc_labels_max_words = 20
        if 'annota-title-fulltext' in configurations:
            self.doc_data_max_words = 180
            self.doc_labels_max_words = 20

        if 'lowercase':
            self.lowercase = True

        if 'vectors/sk/80' in configurations:
            self.embedding_dim = 80
        if 'vectors/sk/200' in configurations:
            self.embedding_dim = 200
        if 'vectors/en/300' in configurations:
            self.embedding_dim = 300

        if 'attention' in configurations:
            self.attention = True

        self.cell_units = 1024
        if 'multilayer' in configurations:
            self.num_layers = 2
        if 'multilayer+' in configurations:
            self.num_layers = 3
        if 'dropout' in configurations:
            self.keep_probability = 0.5
        if 'pro' in configurations:
            self.cell_units = 2048
        if 'eco' in configurations:
            self.cell_units = 512
        if 'eco+' in configurations:
            self.cell_units = 256
        if 'eco++' in configurations:
            self.cell_units = 128

        self.megabatch_size = self.batch_size * 250

