import numpy as np
import os
import pickle

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def batch_generator():
    pass

'''
dataset.pkl
dictionary.pkl
'''

def rerank_batch(data, lens):
    new_lens = np.argsort(lens)[::-1]
    '''output = []
    for i in range(len(data)):
        output.append(data[new_lens[i]])
    return np.array(output), new_lens'''
    return data[new_lens], lens[new_lens]


def rm_padding(data):
    output = []
    lens = np.zeros((data.shape[0],),dtype=np.int)
    for s in range(len(data)):
        for w in range(len(data[s])):
            if data[s][w] == 0:
                w -= 1
                break
        lens[s] = w + 1
        output.append(data[s][:lens[s]])
    #return rerank_batch(np.array(output), lens)
    return lens

def load_sample(sample_path='sample.pkl'):
    with open(sample_path, 'rb') as fd:
        data = pickle.load(fd)
    return data['x1'], data['y1'], data['x2'], data['y2']

def word2onehot(data, vocab_size):
    output = np.zeros((data.shape[0], vocab_size))
    for i in range(data.shape[0]):
        output[i, data[i]] = 1
    return output

def load_corpus(datasetFilePath='trainset/dataset.pkl', train_perc=0.7, sample_prec=0.1):
    corpusFile = open(datasetFilePath, 'rb')
    x_train = np.array(pickle.load(corpusFile))
    y_train = np.array(pickle.load(corpusFile))
    train_size = int(len(x_train)*train_perc*sample_prec)
    total = int(len(x_train)*sample_prec)

    # shuffle the train set
    idx = np.random.permutation(x_train.shape[0])
    x_train = x_train[idx]
    y_train = y_train[idx]
    return x_train[:train_size], y_train[:train_size], \
		x_train[train_size:total], y_train[train_size:total]
    #return x_train[:train_size], y_train[:train_size], x_train[train_size:], y_train[train_size:]

def load_dictionary(dictionaryFilePath='trainset/dictionary.pkl'):
    dictionaryFile = open(dictionaryFilePath, 'rb')
    w2i = pickle.load(dictionaryFile)
    i2w = pickle.load(dictionaryFile)
    return i2w, w2i


'''
small dataset
1.  length:             44 words per tweet
2.  vocabulary size:    14177, 0 for padding word ''
'''

