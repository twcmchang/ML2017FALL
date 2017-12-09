import os
import sys
import argparse
import keras
from six.moves import cPickle
from sklearn.model_selection import train_test_split
from model import MF
from utils import DataLoader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str, default="data/train.csv",
                        help='input train_file')
    parser.add_argument('--test_file', type=str, default="data/test.csv",
                        help='input test_file')
    parser.add_argument('--save_dir', type=str, default='save',
                        help='directory to store checkpointed models')
    parser.add_argument('--dim_embed', type=int, default=100,
                        help='length of user and movie embedding')
    parser.add_argument('--n_epoch', type=int, default=20,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='minibatch size')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
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
    if args.train_file is not None and args.test_file is not None:
        d.read_train_test(train_file=args.train_file,test_file=args.test_file,classification=False)
    else:
        sys.exit("Error! Please make sure you have specify train_file and test_file correctly.")

    args.n_usr = d.n_usr
    args.n_mov = d.n_mov

    if args.init_from is not None:
        if not os.path.exists(os.path.join(args.init_from,"model.h5")):
            sys.exit("Error! model file is not found.")
        md = keras.models.load_model(os.path.join(args.init_from,"model.h5"))
    else:
        md = MF(args).build()

    with open(os.path.join(args.save_dir, 'args.pkl'), 'wb') as f:
        print("Save configuration file.")
        cPickle.dump(args, f)

    print("Training and testing split...")
    X_train, X_val, y_train, y_val = train_test_split(d.train_data, d.train_label, test_size=0.25)

    opt = keras.optimizers.Adam(lr=args.learning_rate)
    md.compile(loss='mse',optimizer=opt)

    checkpoint = keras.callbacks.ModelCheckpoint(os.path.join(args.save_dir,'model.h5'),monitor='val_loss',verbose=1,save_best_only=True,mode='auto')
    csv_logger = keras.callbacks.CSVLogger(os.path.join(args.save_dir,'training.log'))
    earlystop  = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=5, verbose=0, mode='auto')

    md.fit([X_train[:,0],X_train[:,1]],y_train,
           epochs = args.n_epoch,
           validation_data = ([X_val[:,0],X_val[:,1]],y_val),
           callbacks = [checkpoint,csv_logger,earlystop])

if __name__ == '__main__':
    main()
