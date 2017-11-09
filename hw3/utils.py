import pandas as pd
import numpy as np

class DataLoader(object):
    def __init__(self, train_file=None, test_file=None, num_classes = 10):
        if train_file is not None:
            d = pd.read_csv(train_file)
            print('read training dataset: %d records' % d.shape[0])
            x = list()
            y = list()
            for i in range(d.shape[0]):
                tmp = np.reshape(newshape=(48,48), 
                                 a = d['feature'][i].split(' ')).astype('float32')/255
                tmp = np.expand_dims(tmp,axis=-1)
                x.append(tmp)
                y.append(d['label'][i])
                print('working on %d/%d' % (i, d.shape[0]), end='\r')
            self.X_train = np.array(x)
            self.Y_train = self.__one_hot_encoding(np.array(y),num_classes=10)

        if test_file is not None:
            d = pd.read_csv(test_file)
            print('read testing dataset: %d records' % d.shape[0])
            x = list()
            for i in range(d.shape[0]):
                tmp = np.reshape(newshape=(48,48), 
                                 a = d['feature'][i].split(' ')).astype('float32')/255
                tmp = np.expand_dims(tmp,axis=-1)
                x.append(tmp)
                print('working on %d/%d' % (i, d.shape[0]), end='\r')
            self.X_test = np.array(x)
    
    def __one_hot_encoding(self, arr, num_classes):
        res = np.zeros((arr.size, num_classes))
        res[np.arange(arr.size),arr] = 1
        return(res)

class DataGenerator(object):
    # 'Generates data for Keras'
    def __init__(self, list_x, labels, num_classes = 10, shuffle = True):
        self.X = list_x
        self.Y = labels
        self.num_classes = num_classes
        self.shuffle = shuffle
    
    def generate(self,batch_size):
        while 1:
            indexes = np.arange(len(self.X))
            if(self.shuffle==True):
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
                