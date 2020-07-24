from collections import Counter
import itertools
import chardet
import numpy as np
import re
import math


def clean_str(string):
    """
    将文本中的特定字符串做修改和替换处理
    :param string:
    :return:
    """
    string = re.sub(r"[^A-Za-z0-9:(),!?\'\`]", " ", string)
    string = re.sub(r":", " : ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels():
    """
    Loads data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    #folder_prefix = 'data/'
    #x_train = list(open(folder_prefix+"train", 'rb').readlines())
    #x_test = list(open(folder_prefix+"test", 'rb').readlines())
    folder_prefix = 'D:/OneDrive/WORK/datasets/'
    #x_train = list(open(folder_prefix+"20ng-train-all-terms.txt", 'rb').readlines())
    #x_test = list(open(folder_prefix+"20ng-test-all-terms.txt", 'rb').readlines())
    #x_train = list(open(folder_prefix+"webkb-train-stemmed.txt", 'rb').readlines())
    #x_test = list(open(folder_prefix+"webkb-test-stemmed.txt", 'rb').readlines())
    #x_train = list(open(folder_prefix + "20ng-train-stemmed.txt", 'rb').readlines())
    #x_test = list(open(folder_prefix + "20ng-test-stemmed.txt", 'rb').readlines())
    #x_train = list(open(folder_prefix + "20ng-train-no-stop.txt", 'rb').readlines())
    #x_test = list(open(folder_prefix + "20ng-test-no-stop.txt", 'rb').readlines())
    #x_train = list(open(folder_prefix + "r8-train-no-stop.txt", 'rb').readlines())
    #x_test = list(open(folder_prefix + "r8-test-no-stop.txt", 'rb').readlines())
    #x_train = list(open(folder_prefix + "r52-train-all-terms.txt", 'rb').readlines())
    #x_test = list(open(folder_prefix + "r52-test-all-terms.txt", 'rb').readlines())
    #x_train = list(open(folder_prefix + "r52-train-no-stop.txt", 'rb').readlines())
    #x_test = list(open(folder_prefix + "r52-test-no-stop.txt", 'rb').readlines())
    x_train = list(open(folder_prefix + "amazon-reviews-train-no-stop.txt", 'rb').readlines())
    x_test = list(open(folder_prefix + "amazon-reviews-test-no-stop.txt", 'rb').readlines())
    test_size = len(x_test)
    x_text = x_train + x_test

    le = len(x_text)
    for i in range(le):
        encode_type = chardet.detect(x_text[i])
        x_text[i] = x_text[i].decode(encode_type['encoding'])  # 进行相应解码，赋给原标识符（变量
    y = [s.split()[0].split()[0] for s in x_text]
    x_text = [s.split()[1:] for s in x_text]
    #x_text = [clean_str(sent) for sent in x_text]
    #x_text = [s.split()[1:] for s in x_text]

    '''x_text = [clean_str(sent) for sent in x_text]
    y = [s.split(' ')[0].split(':')[0] for s in x_text]
    x_text = [s.split(" ")[1:] for s in x_text]'''
    # Generate labels
    all_label = dict()
    for label in y:
        if label not in all_label:
            all_label[label] = len(all_label) + 1
    one_hot = np.identity(len(all_label))
    y = [one_hot[all_label[label]-1] for label in y]
    return [x_text, y, test_size]


def pad_sentences(sentences, padding_word="<PAD/>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    sequence_length = max(len(x) for x in sentences)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences


def length_distribution(sentences):
    """统计所有句子的长度"""
    sentence_length = [len(x) for x in sentences]
    max_length = max(sentence_length)
    length_col = [0 for x in range(max_length)]
    for length in sentence_length:
        length_col[length-1] += 1
    return length_col


def appropriate_length(length_col, ratio=0.9):
    """计算合适的文本长度"""
    pin_string = []
    for index, i in enumerate(range(len(length_col))):
        for j in range(i):
            pin_string.append(index+1)
    pin = math.ceil(len(pin_string) * ratio)
    return pin_string[pin-1]


def build_vocab(sentences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    # vocabulary_inv=['<PAD/>', 'the', ....]
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    # Mapping from word to index
    # vocabulary = {'<PAD/>': 0, 'the': 1, ',': 2, 'a': 3, 'and': 4, ..}
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]


def build_input_data(sentences, labels, vocabulary):
    """
    Maps sentences and labels to vectors based on a vocabulary.
    """
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    y = np.array(labels)
    return [x, y]


def sampling(sentences, labels, n):
    """
    sample sentences to reduce the size of data
    n, get the 1/n part of sentences
    """
    sentences_s = []
    labels_s = []
    for index in range(0, len(sentences), n):
        sentences_s.append(sentences[index])
        labels_s.append(labels[index])
    return sentences_s, labels_s


def load_data(padding_ratio=0.9):
    """
    Loads and preprocessed data
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    sentences, labels, test_size = load_data_and_labels()
    # sentences, labels = sampling(sentences, labels, 10)
    # test_size = int(np.ceil(test_size / 10))
    length_col = length_distribution(sentences)
    padding_length = appropriate_length(length_col, ratio=padding_ratio)
    sentences_padded = pad_sentences(sentences)
    vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    x, y = build_input_data(sentences_padded, labels, vocabulary)
    return x, y, vocabulary, vocabulary_inv, test_size, length_col, padding_length
