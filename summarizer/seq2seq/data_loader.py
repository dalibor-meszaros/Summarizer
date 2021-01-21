import random

from tools.mysave import pickle_load

import numpy as np


class DataLoader(object):
    def __init__(self, data_path_func, labels_path_func, batch_size, batch_count, type, random_seed):
        self.data_path_func = data_path_func
        self.labels_path_func = labels_path_func
        self.batch_size = batch_size
        self.batch_count = batch_count
        self.type = type

        self.random_seed = random_seed
        if self.random_seed is not None:
            random.seed(self.random_seed)
        self.epochs = 0
        self.megabatch_iterator = None
        self.iterator = None

    def reset(self):
        if self.random_seed is not None:
            random.seed(self.random_seed)
        self.epochs = 0
        self.megabatch_iterator = None
        self.iterator = None

    def next_batch(self):
        if self.iterator is None:
            self.iterator = self.make_iterator()
        batch = next(self.iterator, None)
        if batch is None:
            self.iterator = self.make_iterator()
            batch = next(self.iterator)
            self.epochs += 1
        x = np.array(batch[0]).T
        y = np.array(batch[1]).T
        return x, y

    def make_megabatch_iterator(self):
        batch_indices = list(range(self.batch_count))
        random.shuffle(batch_indices)
        for batch_idx in batch_indices:
            yield batch_idx

    def make_iterator(self):
        if self.megabatch_iterator is None:
            self.megabatch_iterator = self.make_megabatch_iterator()
        megabatch_idx = next(self.megabatch_iterator, None)
        if megabatch_idx is None:
            self.megabatch_iterator = self.make_megabatch_iterator()
            megabatch_idx = next(self.megabatch_iterator, None)

        data_megabatch = pickle_load(self.data_path_func(type=self.type, num=megabatch_idx))
        labels_megabatch = pickle_load(self.labels_path_func(type=self.type, num=megabatch_idx))

        batch_idx = 0
        batch = [[], []]

        indices = list(range(len(data_megabatch)))
        random.shuffle(indices)
        for idx in indices:
            batch[0].append(data_megabatch[idx])
            batch[1].append(labels_megabatch[idx])
            batch_idx += 1
            if batch_idx >= self.batch_size:
                batch_idx = 0
                yield batch
                batch = [[], []]
        return None
