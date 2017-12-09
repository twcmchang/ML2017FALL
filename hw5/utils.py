import os
import re
import random
import numpy as np
from keras.utils import to_categorical

class DataLoader(object):
    def __init__(self):
        super().__init__()
        self.n_mov          = None
        self.n_usr          = None
        self.train_data     = None
        self.train_label    = None
        self.test_data      = None
        
    def read_train_test(self, train_file=None, test_file=None, classification=False):
        if train_file is not None:
            with open(train_file) as f:
                train_data = []
                train_label = []
                _ = f.readline()
                for line in f:
                    line = line.split(",")
                    train_data.append([int(line[1]),int(line[2])])
                    train_label.append(int(line[3])-1) # rating range: [1,5]
            self.train_data  = np.array(train_data)
            if classification:
                self.train_label = to_categorical(train_label)
            else:
                self.train_label = np.array(train_label)

        if test_file is not None:
            with open(test_file) as f:
                test_data = []
                _ = f.readline() # skip the first header line: id,sentence
                for line in f:
                    line = line.split(",")
                    test_data.append([int(line[1]),int(line[2])])
            self.test_data = np.array(test_data)
         
        if train_file is not None or test_file is not None:   
            self.n_usr = max(np.append(self.train_data[:,0], self.test_data[:,0]))+1
            self.n_mov = max(np.append(self.train_data[:,1], self.test_data[:,1]))+1
            print("train_data and train_label obtained.")
            print("there are %d users and %d movies in training dataset." % (self.n_usr, self.n_mov))
        else:
            print("Error! Neither train_file nor test_file is specified.")

    def read_test(self, test_file=None):
        if test_file is not None:
            with open(test_file) as f:
                test_data = []
                _ = f.readline() # skip the first header line: id,sentence
                for line in f:
                    line = line.split(",")
                    test_data.append([int(line[1]),int(line[2])])
            self.test_data = np.array(test_data)
            print("test_data obtained.")


class DataGenerator(object):
    # 'Generates data for Keras'
    def __init__(self, list_x, labels):
        self.X = list_x
        self.Y = labels
    
    def generate(self,batch_size,shuffle=True):
        while 1:
            indexes = np.arange(len(self.X))
            if(shuffle):
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
                