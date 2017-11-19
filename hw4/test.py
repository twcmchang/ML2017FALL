import numpy as np
import pandas as pd
import os
import sys
import time
import keras
import argparse
from six.moves import cPickle
from model import model
from utils import DataLoader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_file', type=str, default=None,
                        help='testing filename')
    parser.add_argument('--output', type=str, default=None,
                        help='output filename')
    parser.add_argument('--config', type=str, default='save/args.pkl',
                        help='configuration file')
    parser.add_argument('--vocab', type=str, default='save/vocab.pkl',
                        help='model file')
    parser.add_argument('--model', type=str, default='save/model.h5',
                        help='model file')
    parser.add_argument('--save_dir', type=str, default='save/',
                        help='save directory')

    args = parser.parse_args()
    test(args)

def test(args):
    print("reading vocabulary...")
    with open(os.path.join(args.save_dir, 'vocab.pkl'), 'rb') as f:
        vocab = cPickle.load(f)

    with open(os.path.join(args.save_dir, 'args.pkl'), 'rb') as f:
        config = cPickle.load(f)

    d = DataLoader()
    if args.test_file is not None:
        d.read_test(test_file=args.test_file)
        d.generate_X_test(maxlen=config.n_word)
    else:
        sys.exit("Error! Please specify your testing file.")

    if args.init_from is not None:
        md = keras.models.load_model(args.init_from)
    else:
        sys.exit("Error! Please specify your model in use.")
    md.summary()

    y_pred = md.predict(d.test_data)
    labels = [np.argmax(y_pred[i]) for i in range(len(y_pred))]
    output = pd.DataFrame({"id":range(len(y_pred)),"label":labels})
    output.to_csv(args.output, index=False)
    print("Testing completed and saved into %s." % args.output)

if __name__ == '__main__':
    main()


