from text_processing.steps.shuffle import TextProcessingShuffler
from text_processing.steps.split import TextProcessingSplitter
from text_processing.steps.word2vec_scrapper import Word2VecScrapper


class TextProcessingRunner(object):
    def __init__(self, configurations, rootdir='./data/', linecount=None, checkpoint=5):
        self.checkpoint = checkpoint
        self.configurations = configurations
        self.rootdir = rootdir
        self.linecount = linecount

    def run(self):
        shuffler = TextProcessingShuffler(configurations=self.configurations, rootdir=self.rootdir, linecount=self.linecount)
        self.linecount = shuffler.linecount
        splitter = TextProcessingSplitter(configurations=self.configurations, rootdir=self.rootdir, linecount=self.linecount)
        w2v_scrapper = Word2VecScrapper(configurations=self.configurations, rootdir=self.rootdir)

        shuffler.shuffle_and_trim(checkpoint=self.checkpoint)
        splitter.make_batches()

        if w2v_scrapper.is_available:
            w2v_scrapper.scrap_word2vecs()

