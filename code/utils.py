from __future__ import print_function

from collections import Counter
import itertools
import numpy as np
import tensorflow as tf
import re
import os
from keras.utils.np_utils import to_categorical
    
# def clean_str(string):
#     """
#     Tokenization/string cleaning.
#     Original from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
#     """
#     string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
#     string = re.sub(r"\'s", " \'s", string)
#     string = re.sub(r"\'ve", " \'ve", string)
#     string = re.sub(r"n\'t", " n\'t", string)
#     string = re.sub(r"\'re", " \'re", string)
#     string = re.sub(r"\'d", " \'d", string)
#     string = re.sub(r"\'ll", " \'ll", string)
#     string = re.sub(r",", " , ", string)
#     string = re.sub(r"!", " ! ", string)
#     string = re.sub(r"\(", " \( ", string)
#     string = re.sub(r"\)", " \) ", string)
#     string = re.sub(r"\?", " \? ", string)
#     string = re.sub(r"\s{2,}", " ", string)
    
#     return string.strip().lower()


def load_data_and_labels_from_many_files(data_folder, data_files):
    """
    Loads sentences from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    print("Loading data...")
    x_text = []
    y = []

    for i, data_file in enumerate(data_files):

        sentences = list(open(data_folder + "/" + data_file, "r").readlines())
        sentences = [s.strip() for s in sentences]
        # Split by words
        # sentences = [clean_str(s) for s in sentences]
        sentences = [s.split() for s in sentences]
        x_text += sentences
        # Labels as numbers
        labels = [i for s in sentences]
        y += labels

    # Generate one-hot labels
    y = to_categorical(y, num_classes=len(data_files))

    return x_text, y

def load_data_and_labels_from_one_file(data_folder, data_file):
    """
    Loads sentences from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    print("Loading data...")
    labels = ["EGY", "GLF", "LAV", "MSA", "NOR"]
    x_text = []
    y = []

    with open(data_folder + "/" + data_file, "r") as f_in:
        for line in f_in:
            sentence, label = line.split("\t")
            # Split by words
            sentence = sentence.strip().split()
            x_text.append(sentence)
            # Labels as numbers
            y.append(labels.index(label.strip()))

    # Generate one-hot labels
    y = to_categorical(y, num_classes=len(labels))

    return x_text, y


def pad_sentences(sentences, padding_word=""):
    """
    Pads all sentences to be the length of the longest sentence.
    Returns padded sentences.
    """
    print("Padding sentences...")
    sequence_length = max(len(x) for x in sentences)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
        
    return padded_sentences


def build_vocab(sentences):
    """
    Builds a vocabulary mapping from token to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    print("Building word vocabulary...")
    word_counts = Counter(itertools.chain(*sentences))
    
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    
    return vocabulary, vocabulary_inv


def build_input_data(sentences, labels, vocabulary):
    """
    Maps sentences and labels to vectors based on a vocabulary.
    """
    print("Converting to ids...")
    x = np.array([
            [vocabulary[word] for word in sentence]
            for sentence in sentences])
    y = np.array(labels)
    
    return x, y


def get_vardial_dataset_from_many_files(data_folder, data_files):
    # Step 1: Read in data
    sentences, labels = load_data_and_labels_from_many_files(data_folder, data_files)

    # Step 2: Pad sentences and convert to ids
    sentences_padded = pad_sentences(sentences)
    vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    x, y = build_input_data(sentences_padded, labels, vocabulary)

    # Step 3: Split train/test set
    # TODO: This is very crude, should use cross-validation
    dev_sample_index = -1 * int(0.1 * float(len(y)))
    x_train, x_dev = x[:dev_sample_index], x[dev_sample_index:]
    y_train, y_dev = y[:dev_sample_index], y[dev_sample_index:]

    vocab_size = len(vocabulary)
    sentence_size = x_train.shape[1]

    print('Train/Dev split: %d/%d' % (len(y_train), len(y_dev)))
    print('train shape:', x_train.shape)
    print('dev shape:', x_dev.shape)
    print('vocab_size', vocab_size)
    print('sentence max words', sentence_size)

    return x_train, x_dev, y_train, y_dev 

def get_vardial_dataset_from_train_dev_file(data_folder, train_file, dev_file):
    # Step 1: Read in data
    sentences_train, labels_train = load_data_and_labels_from_one_file(data_folder, train_file)
    sentences_dev, labels_dev = load_data_and_labels_from_one_file(data_folder, train_file)
    sentences = sentences_train + sentences_dev
    labels = np.concatenate((labels_train, labels_dev))

    # Step 2: Pad sentences and convert to ids
    sentences_padded = pad_sentences(sentences)
    vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    x, y = build_input_data(sentences_padded, labels, vocabulary)

    # Step 3: Split train/test set
    dev_sample_index = len(sentences_train)
    x_train, x_dev = x[:dev_sample_index], x[dev_sample_index:]
    y_train, y_dev = y[:dev_sample_index], y[dev_sample_index:]

    vocab_size = len(vocabulary)
    sentence_size = x_train.shape[1]

    print('Train/Dev split: %d/%d' % (len(y_train), len(y_dev)))
    print('train shape:', x_train.shape)
    print('dev shape:', x_dev.shape)
    print('vocab_size', vocab_size)
    print('sentence max words', sentence_size)

    return x_train, x_dev, y_train, y_dev 

def safe_mkdir(path):
    """ Create a directory if there isn't one already. """
    try:
        os.mkdir(path)
    except OSError:
        pass