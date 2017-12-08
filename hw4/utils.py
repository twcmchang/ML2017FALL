import os
import re
import random
import numpy as np
import gensim
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

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
        self.tokenizer      = None
        self.embedding_matrix = None

    def create_w2v_model(self):
        total_corpus = self.train_sentence+self.test_sentence+self.augment_sentence
        total_corpus = [sent.lower().split(" ") for sent in total_corpus]
        w2v_model = gensim.models.Word2Vec(total_corpus, size=100, window=5, min_count=0, workers=8)
        emb_size  = len(w2v_model["dog"])
        print("embedding size", emb_size)
        print("gensim model vocab size:", len(w2v_model.wv.vocab))

        vocab_size = None
        self.tokenizer = Tokenizer(num_words=vocab_size,filters="\n\t")
        self.tokenizer.fit_on_texts(self.train_sentence+self.test_sentence+self.augment_sentence)
        print("tokenizer obtained.")

        word_index = self.tokenizer.word_index
        print('Found %s unique tokens.' % len(word_index))

        oov_count = 0
        embedding_matrix = np.zeros((len(word_index), emb_size))
        for word, i in word_index.items():
            try:
                embedding_vector = w2v_model.wv[word]
                embedding_matrix[i] = embedding_vector
            except:
                oov_count +=1
                print(word)
        print("oov count: ", oov_count)

        print("embedding matrix obtained.")
        self.embedding_matrix = embedding_matrix
        
    def generate_X_train(self, maxlen):
        train_sentence = self.tokenizer.texts_to_sequences(self.train_sentence)
        self.train_data  = pad_sequences(train_sentence, maxlen=maxlen)
        self.train_label = to_categorical(np.asarray(self.train_label))
        self.train_label = np.array(self.train_label,dtype=int)
        print("train_data and train_label obtained.")

    def generate_X_test(self,maxlen):
        test_sentence = self.tokenizer.texts_to_sequences(self.test_sentence)
        self.test_data = pad_sequences(test_sentence, maxlen=maxlen)
        print("test_data obtained.")

    def generate_X_augment(self,maxlen):
        augment_sentence = self.tokenizer.texts_to_sequences(self.augment_sentence)
        self.augment_data = pad_sequences(augment_sentence,maxlen)

    def augment_X_train(self, maxlen, data , label):
        self.train_data  = np.concatenate((self.train_data , data),axis=0)
        self.train_label = np.concatenate((self.train_label, self.__one_hot_encoding(np.array(label),2)),axis=0)

    def __clean_str(self,string, rm_mark = False):
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`\.\:]", " ", string)
        string = re.sub(r"i ' m","im", string)
        string = re.sub(r"you ' re","youre", string)
        string = re.sub(r"aren ' t","arent", string)
        string = re.sub(r"isn ' t","isnt", string)
        string = re.sub(r"don ' t","dont", string)
        string = re.sub(r"doesn ' t","doesnt", string)
        string = re.sub(r"didn ' t","didnt", string)
        string = re.sub(r"weren ' t","werent", string)
        string = re.sub(r"wasn ' t","wasnt", string)
        string = re.sub(r"won ' t","wont", string)
        string = re.sub(r"wouldn ' t","wouldnt", string)
        string = re.sub(r"shouldn ' t","shouldnt", string)
        string = re.sub(r"can ' t","cant", string)
        string = re.sub(r"ain ' t","aint", string)
        string = re.sub(r"couldn ' t","couldnt", string)
        string = re.sub(r"haven ' t","havent", string)
        string = re.sub(r"hasn ' t","hasnt", string)
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`\.\:]", " ", string)
        if rm_mark:
            string = re.sub(r",", "", string)
            string = re.sub(r"!", "", string)
            string = re.sub(r"\?", "", string)
        return string.lower().strip()
        
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
            self.train_label    = train_label

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
    
