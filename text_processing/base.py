from configuration import ConfigPaths, ConfParams


def _sum_file_lines(file_path):
    with open(file_path, encoding='utf-8') as f:
        return sum(1 for _ in f)


class TextProcessingBase(object):
    def __init__(self, configurations, rootdir, linecount=None):
        self.paths = ConfigPaths(configurations=configurations, rootdir=rootdir)
        self.params = ConfParams(configurations=configurations)
        self.linecount = linecount
        if self.linecount is None:
            self.linecount = self._check_files()

    def _check_files(self):
        try:
            open(self.paths.doc_data_text, encoding='utf-8').close()
        except IOError:
            raise ValueError('Could not open file {}'.format(self.paths.doc_data_text))
        try:
            open(self.paths.doc_labels_text, encoding='utf-8').close()
        except IOError:
            raise ValueError('Could not open file {}'.format(self.paths.doc_labels_text))
        print('Files successfully opened.')
        linecount = _sum_file_lines(self.paths.doc_data_text)
        linecount2 = _sum_file_lines(self.paths.doc_labels_text)
        if linecount != linecount2:
            raise ValueError('Files are not equally long: {} and {}'.format(linecount, linecount2))
        print('Files are equally long: {}'.format(linecount))
        return linecount

    def _check_shuffled_files(self):
        try:
            open(self.paths.doc_data_shuffled, encoding='utf-8').close()
        except IOError:
            raise ValueError('Could not open file {}'.format(self.paths.doc_data_shuffled))
        try:
            open(self.paths.doc_labels_shuffled, encoding='utf-8').close()
        except IOError:
            raise ValueError('Could not open file {}'.format(self.paths.doc_labels_shuffled))
        print('Files successfully opened.')
        linecount = _sum_file_lines(self.paths.doc_data_shuffled)
        linecount2 = _sum_file_lines(self.paths.doc_labels_shuffled)
        if linecount != linecount2:
            raise ValueError('Files are not equally long: {} and {}'.format(linecount, linecount2))
        print('Files are equally long: {}'.format(linecount))
        return linecount
