import itertools
import numpy as np
from math import ceil

from tools import mydict
from tools.mysave import pickle_save
from tools.myprint import print_line
from text_processing.base import TextProcessingBase


def _build_vocab(sentences, vocab=None):
    if vocab is None:
        vocab = dict()
    new_sentences = []
    for sent in sentences:
        new_sent = []
        for word in sent.split(' '):
            if word not in vocab:
                vocab[word] = len(vocab)
            new_sent.append(vocab[word])
        new_sentences.append(new_sent)
    return new_sentences, vocab


class TextProcessingSplitter(TextProcessingBase):
    def _modify_data_keys(self, keys):
        max_keys = self.params.doc_data_max_words
        keys_new = []
        for line in keys:
            line_new = np.zeros(max_keys)
            for idx, key in enumerate(line):
                line_new[idx] = key
            keys_new.append(line_new[::-1])
        return keys_new

    def _modify_lables_keys(self, keys):
        max_keys = self.params.doc_labels_max_words
        keys_new = []
        for line in keys:
            line = line[:max_keys-1]
            line.append(1)
            line_new = np.zeros(max_keys)
            for idx, key in enumerate(line):
                line_new[idx] = key
            keys_new.append(line_new)
        return keys_new

    def _modify_keys(self, keys, type):
        if type == 'data':
            return self._modify_data_keys(keys)
        elif type == 'labels':
            return self._modify_lables_keys(keys)
        raise ValueError('Nonexistent type: {}'.format(type))

    def _split(self):
        train_len = int(self.linecount * 8 / 10)
        valid_len = int((self.linecount - train_len) / 2)
        test_len = int(self.linecount - train_len - valid_len)
        return train_len, valid_len, test_len

    def _make_batches(self, file_in, file_out_batch_func, file_out_vocab, batch_size, minibatch_size, split_lens, type):
        batch_info = {}
        vocab = {
            '__PAD__': 0,
            '__EOS__': 1
        }

        def make_type_batch(batch_type: str, split: int):
            nonlocal batch_info
            nonlocal vocab

            minibatch_count = 0
            batch_count = 0

            with open(file_in, 'r', encoding='utf-8') as f:
                print('Splitting {}'.format(batch_type))
                start = 0
                end = split - 1

                batch = []
                line_idx = 0
                for line in itertools.islice(f, start, end):
                    batch.append(line.strip())
                    line_idx += 1
                    if line_idx >= batch_size:
                        batch_keys, vocab = _build_vocab(batch, vocab)
                        batch_keys = self._modify_keys(batch_keys, type)
                        pickle_save(batch_keys, file_out_batch_func(type=batch_type, num=batch_count))

                        minibatch_count += line_idx / minibatch_size
                        line_idx = 0
                        batch = []
                        batch_count += 1

                batch_keys, vocab = _build_vocab(batch, vocab)
                batch_keys = self._modify_keys(batch_keys, type)
                pickle_save(batch_keys, file_out_batch_func(type=batch_type, num=batch_count))

                minibatch_count = int(minibatch_count + ceil(line_idx / minibatch_size))
                batch_count += 1
                print(
                    'Splitting {} done; saved {} batches/{} minibatches'.format(batch_type, batch_count, minibatch_count))
                batch_info[batch_type] = batch_count
                batch_info['{}_minibatch'.format(batch_type)] = minibatch_count

        print('Splitting file {}.'.format(file_in))
        make_type_batch('train', split_lens[0])
        make_type_batch('valid', split_lens[1])
        make_type_batch('test', split_lens[2])

        print('Saving vocabulary {}.'.format(file_out_vocab))
        pickle_save(mydict.reverse_dict(vocab), file_out_vocab)
        print('Done')
        return batch_info

    def make_batches(self):
        split_lens = self._split()
        batch_info_data = self._make_batches(file_in=self.paths.doc_data_shuffled,
                                             file_out_batch_func=self.paths.batch_data,
                                             file_out_vocab=self.paths.vocab_data,
                                             batch_size=self.params.megabatch_size,
                                             minibatch_size=self.params.batch_size,
                                             split_lens=split_lens,
                                             type='data')
        print_line()
        batch_info_labels = self._make_batches(file_in=self.paths.doc_labels_shuffled,
                                               file_out_batch_func=self.paths.batch_labels,
                                               file_out_vocab=self.paths.vocab_labels,
                                               batch_size=self.params.megabatch_size,
                                               minibatch_size=self.params.batch_size,
                                               split_lens=split_lens,
                                               type='labels')
        print_line()
        batch_info = {
            'data': batch_info_data,
            'labels': batch_info_labels
        }
        pickle_save(batch_info, self.paths.batch_info)
        print('Saved batch info {}.'.format(self.paths.batch_info))
        print_line()
