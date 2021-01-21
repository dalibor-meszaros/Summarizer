class MyLogger(object):
    def __init__(self, silent=False, file=None):
        self.file_path = file
        if self.file_path is not None:
            try:
                self.file = open(self.file_path, 'w', encoding='utf-8')
            except IOError:
                raise ValueError('Could not open file {} for logger.'.format(self.file_path))
        else:
            self.file = None
        self.silent = silent
        self.log = []

    def close(self):
        if self.file is not None and not self.file.closed:
            self.file.close()

    def write(self, string):
        self.log.append(string)
        if not self.silent:
            print(string)
        if self.file is not None:
            self.file.write(string)

    def print_all(self):
        for log in self.log:
            print(log)

    def reset(self, reset_file=False):
        self.log = []
        if reset_file:
            if self.file_path is None:
                raise ValueError('Can not reset file without path given.')
            self.close()
            self.file = open(self.file_path, 'w', encoding='utf-8')
