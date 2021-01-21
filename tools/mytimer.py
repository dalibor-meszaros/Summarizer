import time


class MyTimer:
    def __init__(self):
        self.time_start = time.time()

    def set_time(self, time_new=None):
        if time_new is None:
            self.time_start = time.time()
        else:
            self.time_start = time_new

    def get_elapsed(self):
        return time.time() - self.time_start

    def print_elapsed(self, prefix=None):
        if prefix is None:
            prefix = 'Elapsed time: '
        print('{}{:.2f}s'.format(prefix, (time.time() - self.time_start)))
