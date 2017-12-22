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
        self.n_aux          = 0
        self.train_data     = None
        self.train_aux      = None
        self.train_label    = None
        self.test_aux       = None
        self.test_data      = None

        # optional
        self.attr_usr       = None
        self.attr_mov       = None

    def one_hot_encoding(self,arr, num_classes):
        res = np.zeros((len(arr), num_classes))
        res[np.arange(len(arr)),arr] = 1
        return(res)

    def read_user(self, user_file=None):
        if user_file is not None:
            with open(user_file) as f:
                attr_usr = {}
                _ = f.readline()
                for line in f:
                    line = line.split(sep="::")
                    if line[1] is 'F':
                        line[1] = 0
                    else:
                        line[1] = 1
                    attr_usr[line[0]] = np.array([line[1],line[2],line[3]],dtype=float)
            print("Total number of users: %d" % len(attr_usr))
            self.attr_usr = attr_usr
            self.n_aux += 3

    def read_movie(self, movie_file=None):
        with open(movie_file) as f:
            attr_mov = {}
            mov_type_dict = {}
            _ = f.readline()
            count = 0
            for line in f:
                line = line.split(sep="::")
                year = int(line[1][-5:-1])/1000
                this_type = []
                for ty in line[2].split(sep="|"):
                    if not (ty in mov_type_dict):
                        mov_type_dict[ty] = count
                        this_type.append(count)
                        count = count + 1
                    else:
                        this_type.append(mov_type_dict[ty])
                attr_mov[line[0]] = [year, this_type]
            # [year, hot_encoding_of_types]    
            for key,value in attr_mov.items():
                ft_type = np.sum(self.one_hot_encoding(value[1],count),axis=0)
                attr_mov[key] = np.append(value[0],ft_type/np.sum(ft_type))
            print("There are %d movies in total and can be categorized into %d types." % (len(attr_mov),count))
            self.attr_mov = attr_mov
            self.n_aux += (count+1)

    def read_train_test(self, train_file=None, test_file=None, classification=False):
        if train_file is not None:
            with open(train_file) as f:
                train_data = []
                train_aux = []
                train_label = []
                _ = f.readline()
                for line in f:
                    aux = np.array([])
                    line = re.sub('\s+','',line)
                    line = line.split(",")
                    if self.attr_usr is not None:
                        aux = np.append(aux,self.attr_usr[line[1]])
                    if self.attr_mov is not None:
                        aux = np.append(aux,self.attr_mov[line[2]])
                        train_aux.append(aux)
                    train_data.append([int(line[1]),int(line[2])])
                    train_label.append(int(line[3])) # rating range: [1,5]
            self.train_data = np.array(train_data)
            print(train_data[0])

            if self.attr_usr is not None or self.attr_mov is not None:
                self.train_aux = np.array(train_aux)
                print(train_aux[0])

            if classification:
                self.train_label = to_categorical(train_label-1)
            else:
                self.train_label = np.array(train_label)

        if test_file is not None:
            with open(test_file) as f:
                test_data = []
                test_aux = []
                _ = f.readline() # skip the first header line: id,sentence
                for line in f:
                    aux = np.array([])
                    line = re.sub('\s+','',line)
                    line = line.split(",")
                    if self.attr_usr is not None:
                        aux = np.append(aux,self.attr_usr[line[1]])
                    if self.attr_mov is not None:
                        aux = np.append(aux,self.attr_mov[line[2]])
                        test_aux.append(aux)
                    test_data.append([int(line[1]),int(line[2])])
            self.test_data = np.array(test_data)
            print(test_data[0])
            if self.attr_usr is not None or self.attr_mov is not None:
                self.test_aux = np.array(test_aux)
                print(test_aux[0])
         
        if train_file is not None or test_file is not None:
            self.n_usr = max([self.train_data[i][0] for i in range(len(self.train_data))]+[self.test_data[i][0] for i in range(len(self.test_data))])+1
            self.n_mov = max([self.train_data[i][1] for i in range(len(self.train_data))]+[self.test_data[i][1] for i in range(len(self.test_data))])+1
            #self.n_mov = max(np.append(self.train_data[:,1], self.test_data[:,1]))+1
            print("train_data and train_label obtained.")
            print("there are %d users and %d movies in training dataset." % (self.n_usr, self.n_mov))
        else:
            print("Error! Neither train_file nor test_file is specified.")

    def read_test(self, test_file=None):
        if test_file is not None:
            with open(test_file) as f:
                test_data = []
                test_aux = []
                _ = f.readline() # skip the first header line: id,sentence
                for line in f:
                    aux = np.array([])
                    line = re.sub('\s+','',line)
                    line = line.split(",")
                    if self.attr_usr is not None:
                        aux = np.append(aux,self.attr_usr[line[1]])
                    if self.attr_mov is not None:
                        aux = np.append(aux,self.attr_mov[line[2]])
                        test_aux.append(aux)
                    test_data.append([int(line[1]),int(line[2])])
            self.test_data = np.array(test_data)
            print(test_data[0])
            if self.attr_usr is not None or self.attr_mov is not None:
                self.test_aux = np.array(test_aux)
                print(test_aux[0])


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
                
