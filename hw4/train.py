import tensorflow as tf
import numpy as np
import os
import sys
import time
import argparse
import keras
from six.moves import cPickle
from sklearn.model_selection import train_test_split
from n_model import SentiLSTM
from n_utils import DataLoader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str, default=None,
                        help='input train_file')
    parser.add_argument('--unlabeled_file', type=str, default=None,
                        help='input unlabeled_file')
    parser.add_argument('--test_file', type=str, default=None,
                        help='input test_file')
    parser.add_argument('--save_dir', type=str, default='save',
                        help='directory to store checkpointed models')
    parser.add_argument('--n_word', type=int, default=39,
                        help='maximal number of words in a sentence')
    parser.add_argument('--tokenizer', type=str, default=None,
                        help='pre-trained tokenizer')
    parser.add_argument('--embedding_matrix', type=str, default=None,
                        help='pre-trained embedding_matrix')
    parser.add_argument('--n_epoch', type=int, default=20,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='minibatch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--init_from', type=str, default=None,
                        help='--init_from')

    args = parser.parse_args()
    train(args)

def train(args):
    if not os.path.exists(args.save_dir):
        print("Create directory %s" % args.save_dir)
        os.makedirs(args.save_dir)
    
    d = DataLoader()
    if args.train_file is not None:
        d.read_train(args.train_file)
    else:
        sys.exit("Error! Please at least specify your training file.")
    if args.test_file is not None:
        d.read_test(args.test_file)
    if args.unlabeled_file is not None:
        d.read_augment(args.unlabeled_file)

    if args.tokenizer is not None:
        print("Read tokenizer")
        with open(args.tokenizer, 'rb') as f:
            d.tokenizer = cPickle.load(f)

    if args.embedding_matrix is not None:
        print("Read embedding_matrix")
        with open(args.embedding_matrix, 'rb') as f:
            d.embedding_matrix = cPickle.load(f)
    else:
        print("Create w2v model, tokenizer, embedding_marix")
        d.create_w2v_model()
        with open(os.path.join(args.save_dir,"tokenizer.pkl"), 'wb') as f:
            cPickle.dump(d.tokenizer,f)
        with open(os.path.join(args.save_dir,"embedding_matrix.pkl"), 'wb') as f:
            cPickle.dump(d.embedding_matrix,f)

        args.tokenizer = os.path.join(args.save_dir,"tokenizer.pkl")
        args.embedding_matrix = os.path.join(args.save_dir,"embedding_matrix.pkl")

    if args.init_from is not None:
        if not os.path.exists(os.path.join(args.init_from,"model.h5")):
            sys.exit("Error! model file is not found.")
        md = keras.models.load_model(os.path.join(args.init_from,"model.h5"))
    else:
        md = SentiLSTM(args).build()

    print("Generate X_train and X_test with maximum sentence length = %d" % args.n_word)
    d.generate_X_train(maxlen=args.n_word)
    if args.test_file is not None:
        d.generate_X_test(maxlen=args.n_word)
    if args.unlabeled_file is not None:
        d.generate_X_augment(maxlen=args.n_word)

    with open(os.path.join(args.save_dir, 'args.pkl'), 'wb') as f:
        print("Save configuration file.")
        cPickle.dump(args, f)

    print("Training and testing split...")
    X_train, X_val, y_train, y_val = train_test_split(d.train_data, d.train_label, test_size=0.1)

    opt = keras.optimizers.RMSprop(lr=args.learning_rate, rho=0.9, epsilon=1e-08, decay=0.0)
    md.compile(loss='binary_crossentropy',optimizer=opt,metrics=['accuracy'])

    checkpoint = keras.callbacks.ModelCheckpoint(os.path.join(args.save_dir,'model.h5'),monitor='val_loss',verbose=1,save_best_only=True,mode='auto')
    csv_logger = keras.callbacks.CSVLogger(os.path.join(args.save_dir,'training.log'))
    earlystop  = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=5, verbose=0, mode='auto')
    md.fit(X_train,y_train,
           epochs = args.n_epoch,
           validation_data = (X_val,y_val),
           callbacks = [checkpoint,csv_logger,earlystop])

if __name__ == '__main__':
    main()
