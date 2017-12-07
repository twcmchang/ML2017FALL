# from keras.models import Sequential
from keras.layers import Embedding,LSTM,Dense,Dropout,Bidirectional
from keras.layers import Embedding,Input,InputLayer,Dense,Bidirectional,LSTM,Dropout
from keras.models import Model,load_model
from six.moves import cPickle

class SentiLSTM():
    def __init__(self, args):
        # model parameters
        self.n_word       = args.n_word
        
        if args.w2v_weight is not None:
            with open(args.w2v_weight, 'rb') as f:
                self.w2v_weight = cPickle.load(f)
                print("embedd matrix shape: ",self.w2v_weight.shape)
            self.dim_word_embed = self.w2v_weight.shape[1]
            self.n_vocab        = self.w2v_weight.shape[0]
        else:
            self.n_vocab        = args.n_vocab
            self.dim_word_embed = args.dim_word_embed

    def build(self):
        print('Build model...')
        if self.w2v_weight is not None:
            embedding_layer = Embedding(self.n_vocab,output_dim= self.dim_word_embed,
                                weights=[self.w2v_weight],
                                input_length=self.n_word,
                                trainable=False)
        else:
            embedding_layer = Embedding(self.n_vocab,
                                output_dim= self.dim_word_embed,
                                input_length=self.n_word)
        
        seq_input = Input(shape=(self.n_word,), dtype='int32')
        seq_embedded = embedding_layer(seq_input)

        x = Bidirectional(LSTM(128, activation="tanh", return_sequences=True, dropout=0.3))(seq_embedded)
        x = Bidirectional(LSTM(128, activation="tanh", return_sequences=True, dropout=0.3))(x)
        x = Bidirectional(LSTM(64, activation="tanh", return_sequences=False, dropout=0.3))(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.5)(x)
        output = Dense(2, activation='softmax')(x)
        model = Model(seq_input,output)
        model.summary()
        return(model)
