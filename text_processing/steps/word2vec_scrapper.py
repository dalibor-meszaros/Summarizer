import random

from tools.mysave import pickle_load, numpy_save
from tools.mydict import reverse_dict
from tools.myprint import print_line
from configuration import ConfigPaths, ConfParams

from gensim.models import Word2Vec as w2v
import numpy as np


class Word2VecScrapper(object):
    def __init__(self, configurations, rootdir):
        self.paths = ConfigPaths(configurations=configurations, rootdir=rootdir)
        self.params = ConfParams(configurations=configurations)
        self.is_available = hasattr(self.paths, 'vectors') and \
                            hasattr(self.paths, 'embeddings_data') and \
                            hasattr(self.paths, 'embeddings_labels')
        self.vec_model = None
        self.vec_path = None

    def _scrap_word2vec(self, file_out_embeddings, file_vocab, file_vector):
        if self.vec_model is None or self.vec_path != file_vector:
            self.vec_path = file_vector
            print('Loading gensim model.')
            self.vec_model = w2v.load_word2vec_format(file_vector, binary=True)
            print('Loading done.')
        vec_vocab = reverse_dict({k: v for k, v in enumerate(self.vec_model.index2word)})
        vec_embeddings = self.vec_model.syn0

        print('Loading base vocabulary.')
        vocab = pickle_load(file_vocab)
        print('Loading done.')

        print('Scrapping embeddings.')
        embeddings = [vec_embeddings[vec_vocab['</s>']],
                      vec_embeddings[vec_vocab['</s>']]]
        random.seed(self.params.doc_random_seed)
        for k in range(len(vocab)-2):
            if vocab[k] in vec_vocab:
                idx = vec_vocab[vocab[k]]
                embeddings.append(vec_embeddings[idx])
            else:
                embeddings.append(self.vec_model.seeded_vector(random.Random()))
        numpy_save(np.asanyarray(embeddings, dtype='float32'), file_out_embeddings)
        print('Scrapping done, saved to {}.'.format(file_out_embeddings))

    def scrap_word2vecs(self):
        self._scrap_word2vec(file_out_embeddings=self.paths.embeddings_data,
                             file_vocab=self.paths.vocab_data,
                             file_vector=self.paths.vectors)
        print_line()
        self._scrap_word2vec(file_out_embeddings=self.paths.embeddings_labels,
                             file_vocab=self.paths.vocab_labels,
                             file_vector=self.paths.vectors)
        print_line()
        self.free_memory()

    def free_memory(self):
        del self.vec_model
        self.vec_model = None
        self.vec_path = None
