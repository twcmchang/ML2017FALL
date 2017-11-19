from keras.models import Sequential
from keras.layers import Embedding,LSTM,Dense

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
        model.add(Embedding(self.n_vocab, self.dim_word_embed,input_length = self.n_word))
        model.add(LSTM(self.dim_hidden, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(1, activation='sigmoid'))

        return(model)