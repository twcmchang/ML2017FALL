from keras.models import Sequential
from keras.layers import Embedding,LSTM,Dense,Dropout,Bidirectional

class SentiLSTM():
    def __init__(self, args):
        # model parameters
        self.n_word       = args.n_word
        self.n_vocab      = args.n_vocab
        self.dim_word_embed = args.dim_word_embed 
        self.dim_hidden   = args.dim_hidden

    def build(self):
        print('Build model...')
        model = Sequential()
        model.add(Embedding(self.n_vocab, self.dim_word_embed, input_length = self.n_word))
        model.add(Bidirectional(LSTM(self.dim_hidden, activation="tanh", return_sequences=True),input_shape=(self.n_word, self.dim_word_embed)))
        #model.add(Dropout(0.5))
        model.add(Bidirectional(LSTM(int(self.dim_hidden/2), activation="tanh")))
        model.add(Dropout(0.5))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(2, activation='softmax'))
        return(model)
