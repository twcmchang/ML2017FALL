from keras.layers import Embedding,Input,Dense,Bidirectional,Dropout,Dot,Add,Flatten,Concatenate
from keras.models import Model,load_model
from six.moves import cPickle
from keras import backend as K

class MF():
    def __init__(self, args):
        # model parameters
        self.dim_embed  = args.dim_embed
        self.n_usr      = args.n_usr
        self.n_mov      = args.n_mov
        if args.n_aux > 0:
            self.n_aux  = args.n_aux
        else:
            self.n_aux  = None

    def build(self):
        print('Build model...')
        
        self.usr_embedding_matrix = Embedding(self.n_usr, output_dim = self.dim_embed)
        self.mov_embedding_matrix = Embedding(self.n_mov, output_dim = self.dim_embed)

        self.usr_bias = Embedding(self.n_usr, output_dim = 1, embeddings_initializer='zeros')
        self.mov_bias = Embedding(self.n_mov, output_dim = 1, embeddings_initializer='zeros')

        #self.bias = K.variable(value=0.0)

        usr_input = Input(shape=(1,),dtype='int32')
        usr_embedding = self.usr_embedding_matrix(usr_input)
        usr_embedding = Flatten()(usr_embedding)

        usr_bias = self.usr_bias(usr_input)

        mov_input = Input(shape=(1,),dtype='int32')
        mov_embedding = self.mov_embedding_matrix(mov_input)
        mov_embedding = Flatten()(mov_embedding)

        mov_bias = self.mov_bias(mov_input)

        output = Dot(axes=1)([usr_embedding, mov_embedding])
        output = Add()([output, usr_bias, mov_bias])
        if self.n_aux is not None:
            aux_input = Input(shape=(self.n_aux,), name='aux_input')
            output = Concatenate()([output, aux_input])
        output = Dense(1,activation='linear')(output)   
        # output = Dense(, activation='softmax')(added)
        if self.n_aux is not None:
            model = Model([usr_input,mov_input,aux_input],output)
        else:
            model = Model([usr_input,mov_input],output)
        model.summary()
        return(model)

