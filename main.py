from text_processing.runner import TextProcessingRunner

from summarizer.seq2seq.model import Seq2SeqModel
from summarizer.seq2seq.training import Seq2SeqTraining

"""
Example script to for creating the Text Processing object 
and launching automated text file shuffler and splitter which 
creates multiple loadable batches. Convenient.
"""

# config_name = 'wiki-en-eco'
# config_name = 'annota-sk-title-abstract'
config_text = 'annota-title-fulltext'

config_vectors = ''

# Can be abused to use only part of dataset, improves text processing speed on large files
# linecount = 3554336
# linecount = 50000
linecount = None

# Process text
text_processing = TextProcessingRunner(configurations=[config_text, config_vectors], linecount=linecount)
text_processing.run()

# Create and initialize a seq2seq model, then start training process
model = Seq2SeqModel(configurations=[config_text])
model.initialize()
trainer = Seq2SeqTraining(model)
trainer.run(print_after_steps=20)