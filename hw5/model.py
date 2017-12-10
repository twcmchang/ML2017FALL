from keras.layers import Embedding,Input,Dense,Bidirectional,Dropout,Dot,Add,Flatten
from keras.models import Model,load_model
from six.moves import cPickle
from keras import backend as K

class MF():
    def __init__(self, args):
        # model parameters
        self.dim_embed  = args.dim_embed
        self.n_usr      = args.n_usr
        self.n_mov      = args.n_mov

    def build(self):
        print('Build model...')
        
        self.usr_embedding_matrix = Embedding(self.n_usr, output_dim = self.dim_embed)
        self.mov_embedding_matrix = Embedding(self.n_mov, output_dim = self.dim_embed)

        #self.bias = K.variable(value=0.0)

        usr_input = Input(shape=(1,),dtype='int32')
        usr_embedding = self.usr_embedding_matrix(usr_input)
        usr_embedding = Flatten()(usr_embedding)

        mov_input = Input(shape=(1,),dtype='int32')
        mov_embedding = self.mov_embedding_matrix(mov_input)
        mov_embedding = Flatten()(mov_embedding)

        output = Dot(axes=1)([usr_embedding, mov_embedding])
        output = Dense(1,activation='linear')(output)
        # output = Dense(, activation='softmax')(added)

        model = Model([usr_input,mov_input],output)
        model.summary()
        return(model)

