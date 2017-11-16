import numpy as np
import pandas as pd
import os
import sys
import time
import keras
import argparse
from six.moves import cPickle
from sklearn.model_selection import train_test_split
from model import model
from keras.preprocessing.image import ImageDataGenerator
from utils import DataLoader, DataGenerator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, default=None,
                        help='filename')
    parser.add_argument('--num_classes', type=int, default=7,
                    help='number of classes')
    parser.add_argument('--save_dir', type=str, default='save',
                        help='directory to store checkpointed models')
    parser.add_argument('--n_epoch', type=int, default=200,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='minibatch size')
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='learning rate')
    parser.add_argument('--init_from', type=str, default=None,
                        help='continue training from saved model at this path')

    args = parser.parse_args()
    train(args)

def train(args):
    if args.filename is not None:
        d = DataLoader(train_file=args.filename,num_classes=args.num_classes)
    else:
        sys.exit("Error! Please specify your training file.")
    X_train, X_val, y_train, y_val = train_test_split(d.X_train, d.Y_train, test_size=0.2)

    datagen = ImageDataGenerator(rotation_range=30,width_shift_range=0.2,height_shift_range=0.2,fill_mode='nearest')

    train_generator_aug = datagen.flow(X_train,y_train)

    if args.init_from is not None:
        md = keras.models.load_model(args.init_from)
    else:
        md = model(input_shape=X_train[0].shape,num_classes=args.num_classes)
    md.summary()
    opt = keras.optimizers.Adadelta(lr=0.001, rho=0.95, epsilon=1e-08, decay=0.0)
    md.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])

    checkpoint = keras.callbacks.ModelCheckpoint(os.path.join(args.save_dir,'model.h5'),monitor='val_loss',verbose=1,save_best_only=True,mode='auto')
    csv_logger = keras.callbacks.CSVLogger(os.path.join(args.save_dir,'training.log'))

    md.fit_generator(generator = train_generator_aug,
                    epochs=args.n_epoch,
                    steps_per_epoch = 400,
                    validation_data = (X_val,y_val),
                    callbacks=[checkpoint,csv_logger])

    with open(os.path.join(args.save_dir, 'args.pkl'), 'wb') as f:
        cPickle.dump(args, f)

if __name__ == '__main__':
    main()


