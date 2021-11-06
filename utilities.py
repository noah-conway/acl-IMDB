import numpy as np
#import sklearn
import sys
import os
import random

def tokenize(dir, fname):

    stopwords = [',', 'the', 's', 'I', 'a', 'it', 'for', 'and', 'in', '/><br', '/>a', 'is', 'of', 'as', 'like'] 

    with open(dir + "/" + fname) as f:
        line = [f.readline()]
    words = line[0].split()
    words = [w.lower() for w in words if w not in stopwords]

    return words

def min_count_helper(vocab_list, min_count):
    vocab_unique = list(set(vocab_list))
    if min_count == 1:
        return vocab_unique

    min_count = min_count - 1
    for i in range(min_count):
        for w in vocab_unique:
            vocab_list.remove(w)
            vocab_unique = list(set(vocab_list))
    


    return vocab_unique

def generate_vocab(dir, min_count, max_files):

    pos_dir = dir + "/pos"
    neg_dir = dir + "/neg"

    pos_files = os.listdir(pos_dir)
    neg_files = os.listdir(neg_dir)


    if (max_files != -1):
        num_each = max_files//2 #want equal number of positive and negative samples
        pos_files = pos_files[:num_each]
        neg_files = neg_files[:num_each]

    vocab = []

    for i in range(np.size(pos_files)):
        wordlist = tokenize(pos_dir, pos_files[i])
        vocab = vocab + wordlist

    for i in range(np.size(neg_files)):
        wordlist = tokenize(neg_dir, neg_files[i])
        vocab = vocab + wordlist

    vocab = min_count_helper(vocab, min_count)

    return vocab


def create_word_vector(dir, fname, vocab):
    
    vector = np.zeros(len(vocab))

    wordlist = tokenize(dir, fname)

    for w in wordlist:
        for v in range(len(vocab)):
            if w == vocab[v]:
                vector[v] = vector[v] + 1
            
    return vector

def load_data(dir, vocab, max_files):

    pos_dir = dir + "/pos"
    neg_dir = dir + "/neg"

    pos_files = os.listdir(pos_dir)
    neg_files = os.listdir(neg_dir)

    if (max_files != -1):
        num_each = max_files//2 #want equal number of positive and negative samples
        pos_files = pos_files[:num_each]
        neg_files = neg_files[:num_each]

    features = []
    labels = []
    
    for f in pos_files:
        vec = create_word_vector(pos_dir, f, vocab)
        features.append(vec)
        labels.append(1)

    for f in neg_files:
        vec = create_word_vector(neg_dir, f, vocab)
        features.append(vec)
        labels.append(0)

    return features, labels


    
    
    
