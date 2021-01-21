def reverse_dict(dictionary):
    return dict(zip(dictionary.values(), dictionary.keys()))


def sort_dict_by_key(dictionary):
    return sorted(dictionary)


def sort_dict_by_value(dictionary):
    return sorted(dictionary, key=dictionary.get, reverse=True)


def inc_counter(counter, key):
    if key in counter:
        counter[key] += 1
    else:
        counter[key] = 1


def build_counter(data, counter=None):
    if counter is None:
        counter = dict()
    for ele in data:
        inc_counter(counter, ele)
    return counter


def build_vocab(data, vocab=None):
    if vocab is None:
        vocab = dict()
    for ele in data:
        if ele not in vocab:
            vocab[ele] = len(vocab)
    return vocab


def build_vocab_bidirect(data, vocab=None):
    if vocab is None:
        vocab = dict()
    for ele in data:
        if ele not in vocab:
            vocab[ele] = len(vocab)
    vocab_rev = reverse_dict(vocab)
    return vocab, vocab_rev


def print_dict(dictionary, num_of_prints=0, sort_by_val=False, print_key_first=True):
    if sort_by_val:
        dictionary_sorted = sort_dict_by_value(dictionary)
    else:
        dictionary_sorted = sort_dict_by_key(dictionary)
    if num_of_prints <= 0:
        if print_key_first:
            for key in dictionary_sorted:
                print(key, dictionary[key])
        else:
            for key in dictionary_sorted:
                print(dictionary[key], key)
    else:
        if print_key_first:
            for key in dictionary_sorted[:num_of_prints]:
                print(key, dictionary[key])
        else:
            for key in dictionary_sorted[:num_of_prints]:
                print(dictionary[key], key)


# Simple print into text file for wiki_parser
def print_dict2file(dictionary, file_out=None, lpad=''):
    import sys
    if file_out is None:
        file_out = sys.stdout
    for key in sorted(dictionary):
        file_out.write("{}{}: {}\r\n".format(lpad, key, dictionary[key]))