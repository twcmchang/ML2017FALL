import os
import re
import random
import numpy as np

class DataLoader(object):
    def __init__(self):
        super().__init__()
        self.vocab          = None
        self.vocab_int      = None
        self.train_sentence = None
        self.train_label    = None
        self.test_sentence  = None
        self.train_data     = None
        self.test_data      = None

    def __split_padding_sentence(self, sentence, maxlen):    
        sentence = sentence.lower().split(' ')
        caplen = len(sentence)
        if caplen < maxlen:
            sentence = np.append(np.repeat('<PAD>',maxlen-caplen),sentence)
        else:
            sentence = sentence[:maxlen]
        idx_sentence = []
        for w in sentence: # append sentences
            if w == '<PAD>' or w not in self.vocab:
                idx_sentence.append(0)
            else:
                idx_sentence.append(self.vocab[w])   
        # to numpy array
        idx_sentence = np.asarray(idx_sentence)
        return idx_sentence

    def __get_padding_sentence(self,batch_sentence, maxlen):
        return_sentence = []
        for i in range(len(batch_sentence)):
            idx_sentence = self.__split_padding_sentence(batch_sentence[i], maxlen)
            return_sentence.append(idx_sentence)
        return_sentence = np.vstack(return_sentence).astype(int)
        return return_sentence

    def __clean_str(self,string, rm_mark = False):
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`\.\:]", " ", string)
        #string = re.sub(r"\'s", " is", string)
        string = re.sub(r"\'ve", " have", string)
        string = re.sub(r"n\'t", " not", string)
        string = re.sub(r"\'re", " are", string)
        string = re.sub(r"\'d", " would", string)
        string = re.sub(r"\'ll", " will", string)
        string = re.sub(r"\s{0,},", " ", string)
        string = re.sub(r"\s{2,}", " ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        if rm_mark:
            string = re.sub(r",", " , ", string)
            string = re.sub(r"!", "", string)
            string = re.sub(r"\?", "", string)
        return string.lower().strip()
        
    def __one_hot_encoding(self, arr, num_classes):
        res = np.zeros((arr.size, num_classes))
        res[np.arange(arr.size),arr] = 1
        return(res)

    def read_train(self, train_file=None):
        if train_file is not None:
            with open(train_file) as f:
                train_sentence = []
                train_label = []
                for line in f:
                    line = line.split(" +++$+++ ")
                    train_label.append(int(line[0]))
                    train_sentence.append(self.__clean_str(line[1]))
            self.train_sentence = train_sentence
            self.train_label    = self.__one_hot_encoding(np.array(train_label),2)

    def read_test(self, test_file=None):
        if test_file is not None:
            with open(test_file) as f:
                test_sentence = []
                _ = f.readline() # skip the first header line: id,sentence
                for line in f:
                    line = ''.join(line.split(",")[1:])
                    test_sentence.append(self.__clean_str(line))
            self.test_sentence = test_sentence

    def read_augment(self, augment_file=None):
        if augment_file is not None:
            with open(augment_file) as f:
                augment_sentence = []
                _ = f.readline()
                for line in f:
                    augment_sentence.append(self.__clean_str(line))
            self.augment_sentence = augment_sentence
    
    def set_word_vocab(self,vocab):
        vocab_inv = {}
        for key in vocab.keys():
            vocab_inv[vocab[key]] = key
        self.vocab = vocab
        self.vocab_inv = vocab_inv

    def build_word_vocab(self, top_k_words = 3000): 
        sentences = self.train_sentence + self.test_sentence + self.augment_sentence
        word_counts = {}
        nsents = 0
        for sent in sentences:
            nsents += 1
            sent = self.__clean_str(sent)
            for w in sent.lower().split(' '):
                word_counts[w] = word_counts.get(w, 0) + 1
        del word_counts[""] # removed the empty string
        vocab_tmp = [w for w in sorted(word_counts, key=word_counts.get, reverse=True)]
        print('using top %d' % top_k_words)

        # Build mapping
        vocab_inv = {}
        vocab_inv[0] = '<PAD>'

        vocab = {}
        vocab['<PAD>'] = 0

        idx = 1
        for i in range(top_k_words-1):
            vocab[vocab_tmp[i]] = idx+i
            vocab_inv[idx+i] = vocab_tmp[i]

        self.vocab = vocab
        self.vocab_inv = vocab_inv

    def generate_X_train(self, maxlen):
        self.train_data = self.__get_padding_sentence(self.train_sentence,maxlen)

    def generate_X_test(self,maxlen):
        self.test_data = self.__get_padding_sentence(self.test_sentence,maxlen)

    def generate_X_augment(self,maxlen):
        self.augment_data = self.__get_padding_sentence(self.augment_sentence,maxlen)

    def augment_X_train(self, maxlen, data , label):
        self.train_data  = np.concatenate((self.train_data , data),axis=0)
        self.train_label = np.concatenate((self.train_label, self.__one_hot_encoding(np.array(label),2)),axis=0)

class DataGenerator():
    # 'Generates data for Keras'
    def __init__(self, list_x, labels, shuffle = True):
        self.X = list_x
        self.Y = labels
        self.shuffle = shuffle
    
    def generate(self,batch_size):
        while 1:
            indexes = np.arange(len(self.X))
            if(self.shuffle):
                np.random.shuffle(indexes)
            # Generate batches
            imax = int(len(indexes)/batch_size)
            for i in range(imax):
                # Find list of IDs
                X = [self.X[k] for k in indexes[i*batch_size:(i+1)*batch_size]]
                Y = [self.Y[k] for k in indexes[i*batch_size:(i+1)*batch_size]]
                X = np.array(X)
                Y = np.array(Y)
                yield X, Y
