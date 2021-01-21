from configuration import ConfigPaths, ConfParams
from tools.mysave import pickle_load


class Seq2SeqBase(object):
    def __init__(self, configurations, rootdir='./data/'):
        self.paths = ConfigPaths(configurations=configurations, rootdir=rootdir)
        self.params = ConfParams(configurations=configurations)
        self.batch_info = pickle_load(self.paths.batch_info)