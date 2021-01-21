import random
import linecache

from tools.myprint import print_line
from tools.mystring import remove_punctuation
from text_processing.base import TextProcessingBase


def clean_trim_lower(line, trim, force_lowercase=False):
    line_list = remove_punctuation(line).strip().split(' ')[:trim]
    if force_lowercase:
        for idx, word in enumerate(line_list):
            line_list[idx] = word.lower()
    return ' '.join(line_list) + '\n'


class TextProcessingShuffler(TextProcessingBase):
    def _shuffle_and_trim(self, file_in, file_out, trim, checkpoint):
        random.seed(self.params.doc_random_seed)
        shuffled = list(range(self.linecount))
        random.shuffle(shuffled)
        print('Shuffling file {}.'.format(file_in))
        with open(file_out, 'w', encoding='utf-8') as f:
            check = round((self.linecount / 100) * checkpoint)
            for i, idx in enumerate(shuffled):
                if i % check == 0:
                    print('Shuffle progress {}%.'.format((i / check) * checkpoint))
                f.write(clean_trim_lower(linecache.getline(file_in, idx), trim, force_lowercase=self.params.lowercase))
        linecache.clearcache()
        print('Shuffling done.')

    def shuffle_and_trim(self, checkpoint=5):
        self._shuffle_and_trim(file_in=self.paths.doc_data_text,
                               file_out=self.paths.doc_data_shuffled,
                               trim=self.params.doc_data_max_words,
                               checkpoint=checkpoint)
        print_line()
        self._shuffle_and_trim(file_in=self.paths.doc_labels_text,
                               file_out=self.paths.doc_labels_shuffled,
                               trim=self.params.doc_labels_max_words,
                               checkpoint=checkpoint)
        print_line()
