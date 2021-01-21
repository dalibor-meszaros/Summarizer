import os

from tools.mysave import pickle_load, numpy_load
from tools.myprint import print_memory_usage

from summarizer.seq2seq.base import Seq2SeqBase
from summarizer.seq2seq.data_loader import DataLoader

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import rnn_cell, seq2seq


def dir_size(start_path='.'):
    start_path = os.path.dirname(start_path)
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size


class Seq2SeqModel(Seq2SeqBase):
    initialized = False

    vocab_in = None
    vocab_out = None
    vocab_in_size = None
    vocab_out_size = None
    batch_info = None

    seq_in_len = None
    seq_out_len = None
    cell_units = None
    embedding_dim = None

    # Session specific
    session = None
    saver = None

    # Data iterators
    train_iter = None
    valid_iter = None
    test_iter = None

    # Encoder & Decoder
    encode_in = None
    labels = None

    # Test
    decode_outs_test = None

    # Model operations
    loss = None
    optimizer = None
    train_op = None

    def _initialize_variables_from_configuration(self):
        self.vocab_in = pickle_load(self.paths.vocab_data)
        self.vocab_out = pickle_load(self.paths.vocab_labels)
        self.vocab_in_size = len(self.vocab_in)
        self.vocab_out_size = len(self.vocab_out)
        self.seq_in_len = self.params.doc_data_max_words
        self.seq_out_len = self.params.doc_labels_max_words
        self.cell_units = self.params.cell_units
        self.embedding_dim = self.params.embedding_dim

    def _session_start(self):
        ops.reset_default_graph()
        if self.session is not None:
            self.session.close()
        self.session = tf.InteractiveSession()

    def _prepare_model(self):
        self.encode_in = [tf.placeholder(tf.int32, shape=(None,), name="ei_%i" % i) for i in range(self.seq_in_len)]

        self.labels = [tf.placeholder(tf.int32, shape=(None,), name="l_%i" % i) for i in range(self.seq_out_len)]

        loss_weights = [tf.ones_like(l, dtype=tf.float32) for l in self.labels]

        decode_in = [tf.zeros_like(self.encode_in[0], dtype=np.int32, name="GO")] + self.labels[:-1]

        cell = rnn_cell.GRUCell(self.cell_units)

        if hasattr(self.params, 'keep_probability'):
            self.keep_prob = tf.placeholder("float")
            cell = rnn_cell.DropoutWrapper(cell, output_keep_prob=self.keep_prob)

        if hasattr(self.params, 'num_layers'):
            cell = rnn_cell.MultiRNNCell([cell]*self.params.num_layers)

        with tf.variable_scope("decoders") as scope:
            if self.params.attention:
                decode_outs, decode_state = seq2seq.embedding_attention_seq2seq(encoder_inputs=self.encode_in,
                                                                                decoder_inputs=decode_in,
                                                                                cell=cell,
                                                                                num_encoder_symbols=self.vocab_in_size,
                                                                                num_decoder_symbols=self.vocab_out_size,
                                                                                embedding_size=self.embedding_dim,
                                                                                feed_previous=False)

                scope.reuse_variables()

                self.decode_outs_test, decode_state_test = \
                    seq2seq.embedding_attention_seq2seq(encoder_inputs=self.encode_in,
                                                        decoder_inputs=decode_in,
                                                        cell=cell,
                                                        num_encoder_symbols=self.vocab_in_size,
                                                        num_decoder_symbols=self.vocab_out_size,
                                                        embedding_size=self.embedding_dim,
                                                        feed_previous=True)

            else:
                decode_outs, decode_state = seq2seq.embedding_rnn_seq2seq(encoder_inputs=self.encode_in,
                                                                          decoder_inputs=decode_in,
                                                                          cell=cell,
                                                                          num_encoder_symbols=self.vocab_in_size,
                                                                          num_decoder_symbols=self.vocab_out_size,
                                                                          embedding_size=self.embedding_dim,
                                                                          feed_previous=False)
                scope.reuse_variables()

                self.decode_outs_test, decode_state_test = \
                    seq2seq.embedding_rnn_seq2seq(encoder_inputs=self.encode_in,
                                                  decoder_inputs=decode_in,
                                                  cell=cell,
                                                  num_encoder_symbols=self.vocab_in_size,
                                                  num_decoder_symbols=self.vocab_out_size,
                                                  embedding_size=self.embedding_dim,
                                                  feed_previous=True)

        self.loss = seq2seq.sequence_loss(decode_outs, self.labels, loss_weights, self.vocab_out_size)
        self.optimizer = tf.train.AdamOptimizer(1e-4)
        self.train_op = self.optimizer.minimize(self.loss)

    def _initialize_model(self):
        self.session.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def _overwrite_embeddings_with_w2v(self):
        embeddings_data = numpy_load(self.paths.embeddings_data)
        embeddings_labels = numpy_load(self.paths.embeddings_labels)
        source_embedding_key = "decoders/embedding_rnn_seq2seq/RNN/EmbeddingWrapper/embedding"
        target_embedding_key = "decoders/embedding_rnn_seq2seq/embedding_rnn_decoder/embedding"

        for tf_var in tf.trainable_variables():
            if source_embedding_key in tf_var.name:
                embeddings_in = tf_var
                print('Input embeddings: {}.'.format(tf_var.name))
            elif target_embedding_key in tf_var.name:
                embeddings_out = tf_var
                print('Output embeddings: {}.'.format(tf_var.name))

        self.session.run(embeddings_in.assign(embeddings_data))
        self.session.run(embeddings_out.assign(embeddings_labels))

    def _load_data_iterators(self):
        self.train_iter = DataLoader(data_path_func=self.paths.batch_data,
                                     labels_path_func=self.paths.batch_labels,
                                     batch_size=self.params.batch_size,
                                     batch_count=self.batch_info['data']['train'],
                                     type='train',
                                     random_seed=self.params.doc_random_seed)
        self.valid_iter = DataLoader(data_path_func=self.paths.batch_data,
                                     labels_path_func=self.paths.batch_labels,
                                     batch_size=self.params.batch_size,
                                     batch_count=self.batch_info['data']['valid'],
                                     type='valid',
                                     random_seed=self.params.doc_random_seed)
        self.test_iter = DataLoader(data_path_func=self.paths.batch_data,
                                    labels_path_func=self.paths.batch_labels,
                                    batch_size=self.params.batch_size,
                                    batch_count=self.batch_info['data']['test'],
                                    type='test',
                                    random_seed=self.params.doc_random_seed)

    def initialize(self):
        self._initialize_variables_from_configuration()
        print('Initialized variables from predefined configurations.')
        print_memory_usage()

        self._session_start()
        print('Session started.')
        print_memory_usage()

        self._prepare_model()
        print('Model variables defined.')
        print_memory_usage()

        self._initialize_model()
        print('Model variables initialized.')
        print_memory_usage()

        if hasattr(self.paths, 'embeddings_data') and hasattr(self.paths, 'embeddings_labels'):
            self._overwrite_embeddings_with_w2v()
            print('Embeddings overwritten.')
            print_memory_usage()

        self._load_data_iterators()
        print('Data set iterators created.')
        print_memory_usage()

        self.initialized = True
        print('Initialization complete!')

    def load(self, path):
        if not self.initialized:
            self.initialize()
        self.saver.restore(self.session, path)
        print('Suffering restored.')

    def save(self, path):
        if not self.initialized:
            raise ValueError('Trying to save uninitialized mode!')
        save_path = self.saver.save(self.session, path)
        print('Sanity saved in file: %s' % save_path)
        print('Size: %.3f MB' % (dir_size(save_path) / 1000000.))
