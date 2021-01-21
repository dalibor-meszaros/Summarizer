import sys

from collections import OrderedDict

from tools.mytimer import MyTimer
from tools.mysave import pickle_save, pickle_load
from tools.myprint import print_line
from tools.mystring import remove_punctuation
from tools.mydict import reverse_dict

from summarizer.seq2seq.model import Seq2SeqModel
from summarizer.seq2seq.data_loader import DataLoader

import numpy as np
import nltk.translate.bleu_score as bleu
import matplotlib.pyplot as plt

from IPython.display import clear_output


class Seq2SeqTraining(object):
    timer = None
    timer_total = None
    graph_data = None

    def __init__(self, seq2seq_model: Seq2SeqModel):
        self.model = seq2seq_model

        if not self.model.initialized:
            raise ValueError('Using uninitialized model for training!')

    def get_feed(self, x, y):
        feed_dict = {self.model.encode_in[t]: x[t] for t in range(self.model.seq_in_len)}
        feed_dict.update({self.model.labels[t]: y[t] for t in range(self.model.seq_out_len)})
        return feed_dict

    def train_batch(self, data_iter: DataLoader):
        x, y = data_iter.next_batch()
        feed_dict = self.get_feed(x, y)
        if hasattr(self.model.params, 'keep_probability'):
            feed_dict[self.model.keep_prob] = self.model.params.keep_probability
        _, out = self.model.session.run([self.model.train_op, self.model.loss], feed_dict)
        return out

    def get_eval_batch_data(self, data_iter: DataLoader):
        x, y = data_iter.next_batch()
        feed_dict = self.get_feed(x, y)
        if hasattr(self.model.params, 'keep_probability'):
            feed_dict[self.model.keep_prob] = 1.
        all_out = self.model.session.run([self.model.loss] + self.model.decode_outs_test, feed_dict)
        eval_loss = all_out[0]
        decode_out = np.array(all_out[1:]).transpose([1, 0, 2])
        return eval_loss, decode_out, x, y

    def eval_batch(self, data_iter: DataLoader, num_batches):
        def get_bleu_non0_lists(list_a, list_b):
            for idx, val in enumerate(list_a):
                if val == 1:
                    pos = idx
                    break
            return list_a[:pos], list_b[:pos]

        accuracy = {
            'loss': list(),
            'perfect': list(),
            'bleu1_non0': list(),
            'bleu1': list(),
            'bleu4': list(),
            'bleu_half': list()
        }
        accuracy = OrderedDict(sorted(accuracy.items()))

        for i in range(num_batches):
            eval_loss, output, x, y = self.get_eval_batch_data(data_iter)
            accuracy['loss'].append(eval_loss)

            for idx in range(len(output)):
                real = y.T[idx]
                predict = np.argmax(output, axis=2)[idx]

                accuracy['perfect'].append(all([True if r == p else False for r, p in zip(real, predict)]))

                real_n0, predict_n0 = get_bleu_non0_lists(real, predict)
                accuracy['bleu1_non0'].append(
                    float(bleu.modified_precision([real_n0], predict_n0, n=1)))
                accuracy['bleu1'].append(
                    float(bleu.modified_precision([real], predict, n=1)))
                accuracy['bleu4'].append(
                    float(bleu.modified_precision([real], predict, n=4)))
                accuracy['bleu_half'].append(
                    float(bleu.modified_precision([real], predict, n=self.model.seq_out_len / 2)))

        for k, v in accuracy.items():
            accuracy[k] = np.mean(v)

        return accuracy

    def load(self, path):
        variables = pickle_load(path)
        self.timer = variables['timer']
        self.timer_total = variables['timer_total']
        self.graph_data = variables['graph_data']
        print('Variables restored.')

    def save(self, path):
        variables = {
            'timer': self.timer,
            'timer_total': self.timer_total,
            'graph_data': self.graph_data,
            'configuration': {
                'paths': self.model.paths,
                'params': self.model.params
            }
        }
        pickle_save(variables, path)
        print('Variables saved in file: %s.' % path)

    def print_graphs(self):
        plt.plot(self.graph_data['TrainSet_Loss'])
        plt.plot(self.graph_data['ValidSet_Loss'])
        plt.legend(['TrainSet_Loss', 'ValidSet_Loss'], loc='upper right')
        plt.show()

        legend = []
        for k, v in self.graph_data.items():
            if 'Loss' not in k and 'TrainSet' in k:
                plt.plot(v)
                legend.append(k)
        plt.legend(legend, loc='upper left')
        plt.show()

        legend = []
        for k, v in self.graph_data.items():
            if 'Loss' not in k and 'ValidSet' in k:
                plt.plot(v)
                legend.append(k)
        plt.legend(legend, loc='upper left')
        plt.show()

    @staticmethod
    def print_accuracy(accuracy, label):
        print('{} set:'.format(label.capitalize()))
        for k, v in accuracy.items():
            if k == 'loss':
                print('{} = {:.2f}'.format(k, v))
            else:
                print('{} = {:.2f}%'.format(k, v * 100))

    def print_examples(self, count, x, y, o, file=None):
        if file is None:
            file = sys.stdout
        print(file=file)
        print('Input sentences:', file=file)
        for idx in range(count):
            print([int(word_idx) for word_idx in reversed(x.T[idx].tolist())], file=file)
            print(self._indices_to_sentence_clean(reversed(x.T[idx].tolist()), self.model.vocab_in), file=file)

        print(file=file)
        print('True sentences:', file=file)
        for idx in range(count):
            print([int(word_idx) for word_idx in y.T[idx].tolist()], file=file)
            print(self._indices_to_sentence_clean(y.T[idx], self.model.vocab_out), file=file)

        print(file=file)
        print('Predicted sentences:', file=file)
        for idx in range(count):
            print(np.argmax(o, axis=2)[idx], file=file)
            print(self._indices_to_sentence_clean(np.argmax(o, axis=2)[idx], self.model.vocab_out), file=file)

    @staticmethod
    def _indices_to_sentence_clean(indices, idx2word, banned_indices=None):
        if banned_indices is None:
            banned_indices = [0, 1]  # PAD, EOS
        sentence = []
        for idx in indices:
            if idx not in banned_indices:
                sentence.append(idx2word[idx])
        return ' '.join(sentence)

    def test(self,
             custom_batch_count=None,
             disable_accuracy=False,
             disable_examples=False):
        print('Evaluating test...')
        print_line()
        timer = MyTimer()
        timer.set_time()
        num_batches = self.model.batch_info['data'][
            'test_minibatch'] if custom_batch_count is None else custom_batch_count
        test_accuracy = self.eval_batch(self.model.test_iter, num_batches)
        eval_loss, output, x, y = self.get_eval_batch_data(self.model.test_iter)

        clear_output()
        print('Evaluation of test complete')
        timer.print_elapsed()
        timer.set_time()
        print_line()

        if not disable_accuracy:
            self.print_accuracy(test_accuracy, 'Test')
            print()
        if not disable_examples:
            self.print_examples(count=15, x=x, y=y, o=output)
        timer.print_elapsed()
        print_line()

    def test_sentence(self, text):
        vocab = reverse_dict(self.model.vocab_in)  # we have idx2word but missing word2idx dictionary

        word_list = remove_punctuation(text).strip().split(' ')[:self.model.params.doc_data_max_words]
        if self.model.params.lowercase:
            for idx, word in enumerate(word_list):
                word_list[idx] = word.lower()
        key_list = np.zeros(self.model.params.doc_data_max_words)
        for idx, word in enumerate(word_list):
            if word in vocab:
                key_list[idx] = vocab[word]
            else:
                key_list[idx] = 0
        key_list = key_list[::-1]
        dummy = [0] * self.model.params.doc_labels_max_words  # we need to pass something as output example
        x = np.array([key_list]).T
        y = np.array([dummy]).T

        feed_dict = self.get_feed(x, y)
        all_out = self.model.session.run([self.model.loss] + self.model.decode_outs_test, feed_dict)
        decode_out = np.array(all_out[1:]).transpose([1, 0, 2])

        print('Input sentence:')
        print([int(word_idx) for word_idx in reversed(x.T[0].tolist())])
        print(self._indices_to_sentence_clean(reversed(x.T[0].tolist()), self.model.vocab_in))
        print()
        print('Predicted sentence:')
        print(np.argmax(decode_out, axis=2)[0])
        print(self._indices_to_sentence_clean(np.argmax(decode_out, axis=2)[0], self.model.vocab_out))

    def run(self,
            print_after_steps=60,
            custom_range=100000,
            disable_accuracy=False,
            disable_examples=False,
            disable_graphs=False,
            output_file=None):
        self.timer = MyTimer() if self.timer is None else self.timer
        self.timer_total = MyTimer() if self.timer_total is None else self.timer_total
        self.graph_data = OrderedDict() if self.graph_data is None else self.graph_data

        print('Start!')
        print_line()
        for i in range(custom_range):  # please interrupt manually, may take a while
            try:
                _ = self.train_batch(self.model.train_iter)
                if i % print_after_steps == 0:
                    print('Evaluating...')

                    self.timer.set_time()
                    validation_accuracy = self.eval_batch(self.model.valid_iter, 1)
                    train_accuracy = self.eval_batch(self.model.train_iter, 1)
                    eval_loss, output, x, y = self.get_eval_batch_data(self.model.valid_iter)

                    clear_output()
                    print('Evaluation complete')
                    self.timer.print_elapsed()
                    self.timer.set_time()
                    print_line()

                    if not disable_accuracy:
                        self.print_accuracy(validation_accuracy, 'Validation')
                        print()
                        self.print_accuracy(train_accuracy, 'Training')
                        print()

                    # ==================================================================================================

                    if not disable_examples:
                        self.print_examples(count=3, x=x, y=y, o=output)

                    if output_file is not None:
                        with open(output_file, 'a', encoding='utf-8') as f:
                            self.print_examples(count=3, x=x, y=y, o=output, file=f)
                            print(file=f)
                            print_line(file=f)
                            print(file=f)

                    # ==================================================================================================

                    if not any('ValidSet' in k for k in self.graph_data.keys()):
                        for k, v in validation_accuracy.items():
                            self.graph_data['ValidSet_{}'.format(k.capitalize())] = [v]
                    else:
                        for k, v in validation_accuracy.items():
                            self.graph_data['ValidSet_{}'.format(k.capitalize())].append(v)

                    if not any('TrainSet' in k for k in self.graph_data.keys()):
                        for k, v in train_accuracy.items():
                            self.graph_data['TrainSet_{}'.format(k.capitalize())] = [v]
                    else:
                        for k, v in train_accuracy.items():
                            self.graph_data['TrainSet_{}'.format(k.capitalize())].append(v)

                    if not disable_graphs:
                        self.print_graphs()

                    # ==================================================================================================

                    self.timer.print_elapsed()
                    self.timer_total.print_elapsed('Total elapsed time: ')
                    print_line()
                    self.timer.set_time()
            except KeyboardInterrupt:
                print_line()
                self.timer_total.print_elapsed('Total elapsed time: ')
                print('Interrupted by user!')
                break

