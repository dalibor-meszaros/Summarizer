import string


def remove_punctuation(text, punctuation=None):
    if punctuation is None:
        punctuation = string.punctuation + u'„“–ʾ†-/'  # common symbols in documents
    for p in punctuation:
        text = text.replace(p, ' ')
    return ' '.join(text.split()).strip()
