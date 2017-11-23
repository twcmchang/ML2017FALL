import tensorflow as tf
import numpy as np
import os
import sys
import time
import argparse
import keras
from six.moves import cPickle
from sklearn.model_selection import train_test_split
from model import SentiLSTM
from utils import DataLoader, DataGenerator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str, default='data/training_label.txt',
                        help='directory to store checkpointed models')
    parser.add_argument('--test_file', type=str, default='data/testing_data.txt',
                        help='directory to store checkpointed models')
    parser.add_argument('--save_dir', type=str, default='save',
                        help='directory to store checkpointed models')
    parser.add_argument('--dim_word_embed', type=int, default=200,
                        help='dimension of word embedding')    
    parser.add_argument('--dim_hidden', type=int, default=1000,
                        help='dimension of LSTM hidden state')
    parser.add_argument('--n_word', type=int, default=20,
                        help='maximal number of words in a sentence')
    parser.add_argument('--n_vocab', type=int, default=4000,
                        help='vocabulary size')
    parser.add_argument('--n_epoch', type=int, default=100,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='minibatch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--init_from', type=str, default=None,
                        help='--init_from')

    args = parser.parse_args()
    train(args)

def train(args):

    d = DataLoader()
    if args.train_file is not None:
        d.read_train(args.train_file)
    else:
        sys.exit("Error! Please specify your training file.")

    if args.test_file is not None:
        d.read_test(args.test_file)
    else:
        sys.exit("Error! Please specify your testing file.")
    
    with open(os.path.join(args.save_dir, 'len'+str(args.n_word)+'_size'+str(args.n_vocab)+'_lstm'+str(args.dim_hidden)+'_wemb'+str(args.dim_word_embed)+'_args.pkl'), 'wb') as f:
        cPickle.dump(args, f)

    print("Build vocabulary using the most frequent top %d words..." % args.n_vocab )
    d.build_word_vocab(top_k_words = args.n_vocab)

    with open(os.path.join(args.save_dir, 'size'+str(args.n_vocab)+'_vocab.pkl'), 'wb') as f:
    	cPickle.dump(obj = d.vocab,file=f)

    print("Generate X_train and X_test with maximum sentence length = %d" % args.n_word)
    d.generate_X_train(maxlen=args.n_word)
    d.generate_X_test(maxlen=args.n_word)

    print("Training and testing split...")
    X_train, X_val, y_train, y_val = train_test_split(d.train_data, d.train_label, test_size=0.2)

    print("Create data generators...")
    train_gen = DataGenerator(X_train, y_train)
    val_gen  = DataGenerator(X_val, y_val)

    if args.init_from is not None:
        md = keras.models.load_model(args.init_from)
    else:
        md = SentiLSTM(args).build()
    md.summary()
    opt = keras.optimizers.RMSprop(lr=args.learning_rate, rho=0.9, epsilon=1e-08, decay=0.0)
    md.compile(loss='binary_crossentropy',optimizer=opt,metrics=['accuracy'])

    checkpoint = keras.callbacks.ModelCheckpoint(os.path.join(args.save_dir,'lstm'+str(args.dim_hidden)+'_wemb'+str(args.dim_word_embed)+'_model.h5'),monitor='val_loss',verbose=1,save_best_only=True,mode='auto')
    csv_logger = keras.callbacks.CSVLogger(os.path.join(args.save_dir,'lstm'+str(args.dim_hidden)+'_wemb'+str(args.dim_word_embed)+'_training.log'))

    md.fit_generator(generator = train_gen.generate(args.batch_size),
                    epochs=args.n_epoch,
                    steps_per_epoch = 400,
                    validation_data = (X_val,y_val),
                    callbacks=[checkpoint,csv_logger])

if __name__ == '__main__':
    main()


