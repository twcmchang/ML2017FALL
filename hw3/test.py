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
from utils import DataLoader, DataGenerator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, default=None,
                        help='testing filename')
    parser.add_argument('--output', type=str, default=None,
                        help='output filename')
    parser.add_argument('--num_classes', type=int, default=7,
                    help='number of classes')
    parser.add_argument('--init_from', type=str, default='save/model.h5',
                        help='continue training from saved model at this path')

    args = parser.parse_args()
    test(args)

def test(args):
    if args.filename is not None:
        d = DataLoader(test_file=args.filename,num_classes=args.num_classes)
    else:
        sys.exit("Error! Please specify your testing file.")

    if args.init_from is not None:
        md = keras.models.load_model(args.init_from)
    else:
        sys.exit("Error! Please specify your model in use.")
    md.summary()
    #adam = keras.optimizers.Adam(lr=args.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    #md.compile(loss='categorical_crossentropy',optimizer=adam,metrics=['accuracy'])

    y_pred = md.predict(d.X_test)
    labels = [np.argmax(y_pred[i]) for i in range(len(y_pred))]
    output = pd.DataFrame({"id":range(len(y_pred)),"label":labels})
    output.to_csv(args.output, index=False)
    print("Testing completed and saved into %s." % args.output)

if __name__ == '__main__':
    main()


