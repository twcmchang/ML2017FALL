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
    parser.add_argument('--train_file', type=str, default=None,
                        help='input train_file')
    parser.add_argument('--unlabeled_file', type=str, default=None,
                        help='input unlabeled_file')
    parser.add_argument('--test_file', type=str, default=None,
                        help='input test_file')
    parser.add_argument('--save_dir', type=str, default='save',
                        help='directory to store checkpointed models')
    parser.add_argument('--dim_word_embed', type=int, default=200,
                        help='dimension of word embedding')
    parser.add_argument('--dim_hidden', type=int, default=1000,
                        help='dimension of LSTM hidden state')
    parser.add_argument('--n_word', type=int, default=20,
                        help='maximal number of words in a sentence')
    parser.add_argument('--n_vocab', type=int, default=80000,
                        help='vocabulary size')
    parser.add_argument('--n_epoch', type=int, default=100,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='minibatch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--init_from', type=str, default=None,
                        help='--init_from')
    parser.add_argument('--self_training_round', type=int, default=0,
                        help='self training or not')

    args = parser.parse_args()
    
    if not args.self_training:
        print("training from scratch!")
        train(args)
    else:
        print("self training mode!")
        self_train(args)

    args = parser.parse_args()
    train(args)

def train(args):
    d = DataLoader()
    if args.train_file is not None:
        d.read_train(args.train_file)
    else:
        sys.exit("Error! Please at least specify your training file.")
    if args.test_file is not None:
        d.read_test(args.test_file)
    if args.unlabeled_file is not None:
        d.read_augment(args.unlabeled_file)

    if args.init_from is not None:
        if not os.path.exists(args.init_from,"model.h5"):
            sys.exit("Error! model file is not found.")
            
        if not os.path.exists(args.init_from,"args.pkl"):
            sys.exit("Error! configuration file is not found.")
            
        if not os.path.exists(args.init_from,"vocab.pkl"):
            sys.exit("Error! vocabulary file is not found.")
            
        md = keras.models.load_model(os.path.join(args.init_from,"model.h5"))
        with open(os.path.join(args.init_from,"args.pkl"), 'rb') as f:
            config = cPickle.load(f)
            args.n_word     = config.n_word
            args.n_vocab    = confif.n_vocab
            args.dim_hidden = config.dim_hidden
            args.dim_word_embed = config.dim_word_embed
        with open(os.path.join(args.init_from,"vocab.pkl"), 'rb') as f:
            d.set_existing_vocab(f)

    else:
        md = SentiLSTM(args).build()
        print("Build vocabulary using the most frequent top %d words..." % args.n_vocab )
        d.build_word_vocab(top_k_words = args.n_vocab)

    print("Generate X_train and X_test with maximum sentence length = %d" % args.n_word)
    d.generate_X_train(maxlen=args.n_word)
    if args.test_file is not None:
        d.generate_X_test(maxlen=args.n_word)
    if args.unlabeled_file is not None:
        d.generate_X_augment(maxlen=args.n_word)

    if not os.path.exists(args.save_dir):
        print("Create directory %s" % args.save_dir)
        os.makedirs(args.save_dir)
    
    with open(os.path.join(args.save_dir, 'args.pkl'), 'wb') as f:
        print("Save configuration file.")
        cPickle.dump(args, f)

    with open(os.path.join(args.save_dir, 'vocab.pkl'), 'wb') as f:
        print("Save vocabulary file.")
        cPickle.dump(obj = d.vocab,file=f)

    print("Training and testing split...")
    X_train, X_val, y_train, y_val = train_test_split(d.train_data, d.train_label, test_size=0.2)

    md.summary()
    if args.init_from is None:
        print("Create data generators...")
        train_gen = DataGenerator(X_train, y_train)
        
        opt = keras.optimizers.RMSprop(lr=args.learning_rate, rho=0.9, epsilon=1e-08, decay=0.0)
        md.compile(loss='binary_crossentropy',optimizer=opt,metrics=['accuracy'])

        checkpoint = keras.callbacks.ModelCheckpoint(os.path.join(args.save_dir,'lstm'+str(args.dim_hidden)+'_wemb'+str(args.dim_word_embed)+'_model.h5'),monitor='val_loss',verbose=1,save_best_only=True,mode='auto')
        csv_logger = keras.callbacks.CSVLogger(os.path.join(args.save_dir,'lstm'+str(args.dim_hidden)+'_wemb'+str(args.dim_word_embed)+'_training.log'))
        earlystop  = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5, verbose=0, mode='auto')

        md.fit_generator(generator = train_gen.generate(args.batch_size),
                        epochs=args.n_epoch,
                        steps_per_epoch = 400,
                        validation_data = (X_val,y_val),
                        callbacks=[checkpoint,csv_logger,earlystop])

    if args.self_training_round > 0:
        def one_hot_encoding(self, arr, num_classes):
            res = np.zeros((arr.size, num_classes))
            res[np.arange(arr.size),arr] = 1
            return(res)

        print("===== self training phrase =====")

        for r in range(args.self_training_round):
            print("%d round..." % r)
            y_pred = md.predict(d.augment_data)
            i_aug = [i for i in range(len(y_pred)) if np.max(y_pred[i]) >= 0.9]
            print("add %d sentences using threshold = 0.9" % len(i_aug))
            labels = [np.argmax(y_pred[i]) for i in i_aug]
            X_train = np.concatenate((X_train , d.augment_data[i_aug]),axis=0)
            y_train = np.concatenate((y_train , one_hot_encoding(np.array(labels),2)),axis=0)

            train_gen = DataGenerator(X_train, y_train)

            checkpoint = keras.callbacks.ModelCheckpoint(os.path.join(args.save_dir,str(r)+'_lstm'+str(args.dim_hidden)+'_wemb'+str(args.dim_word_embed)+'_model.h5'),monitor='val_loss',verbose=1,save_best_only=True,mode='auto')
            csv_logger = keras.callbacks.CSVLogger(os.path.join(args.save_dir,str(r)+'_lstm'+str(args.dim_hidden)+'_wemb'+str(args.dim_word_embed)+'_training.log'))
            earlystop  = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5, verbose=0, mode='auto')

            md.fit_generator(generator = train_gen.generate(args.batch_size),
                            epochs=args.n_epoch,
                            steps_per_epoch = 400,
                            validation_data = (X_val,y_val),
                            callbacks=[checkpoint,csv_logger,earlystop])
            # labels = [np.argmax(y_pred[i]) for i in range(len(y_pred))]
        



# def self_train(args):
#     if args.self_training is True:
#         if (args.init_from is not None) and (args.unlabeled_file is not None):
#             if not os.path.exists(args.init_from,"model.h5"):
#                 sys.exit("Error! model file is not found.")

#             if not os.path.exists(args.init_from,"args.pkl"):
#                 sys.exit("Error! configuration file is not found.")

#             if not os.path.exists(args.init_from,"vocab.pkl"):
#                 sys.exit("Error! vocabulary file is not found.")

#             print("loaded pre-trained model from %s" % os.path.join(args.init_from,"model.h5"))
#             md = keras.models.load_model(os.path.join(args.init_from,"model.h5"))
            
#             print("setting hyper-parameters from %s" % os.path.join(args.init_from,"args.pkl"))
#             with open(os.path.join(args.init_from,"args.pkl"), 'rb') as f:
#                 config = cPickle.load(f)
#             args.n_word     = config.n_word
#             args.n_vocab    = confif.n_vocab
#             args.dim_hidden = config.dim_hidden
#             args.dim_word_embed = config.dim_word_embed
            
#             print("setting vocabulary from %s" % os.path.join(args.init_from,"vocab.pkl"))
#             with open(os.path.join(args.init_from,"vocab.pkl"), 'rb') as f:
#                 vocab = cPickle.load(f)
#             d.set_existing_vocab(vocab)
            
#         else:
#             sys.exit("Self training mode cannot be without pre-trained model. Or make sure the unlabeled file is given.")
    

if __name__ == '__main__':
    main()