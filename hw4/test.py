import numpy as np
import pandas as pd
import os
import sys
import time
import keras
import argparse
from six.moves import cPickle
from model import SentiLSTM
from utils import DataLoader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_file', type=str, default=None,
                        help='testing filename')
    parser.add_argument('--output', type=str, default=None,
                        help='output filename')
    parser.add_argument('--save_dir', type=str, default='save/',
                        help='save directory')

    args = parser.parse_args()
    test(args)

def test(args):
    d = DataLoader()
    if os.path.exists(os.path.join(args.save_dir, "tokenizer.pkl")):
        print("Read tokenizer")
        with open(os.path.join(args.save_dir, "tokenizer.pkl"), 'rb') as f:
            d.tokenizer = cPickle.load(f)
    else:
        sys.exit("Error! Please put your tokenizer in %s." % args.save_dir)

    if os.path.exists(os.path.join(args.save_dir, "embedding_matrix.pkl")):
        print("Read embedding_matrix")
        with open(os.path.join(args.save_dir, "embedding_matrix.pkl"), 'rb') as f:
            d.embedding_matrix = cPickle.load(f)
    else:
        sys.exit("Error! Please put your embedding_matrix in %s." % args.save_dir)
    
    if os.path.exists(os.path.join(args.save_dir,"args.pkl")):
        with open(os.path.join(args.save_dir,"args.pkl"), 'rb') as f:
            config = cPickle.load(f)
    else:
        sys.exit("Error! Please put your args in %s" % args.save_dir)

    if args.test_file is not None:
        d.read_test(test_file=args.test_file)
        d.generate_X_test(maxlen=config.n_word)
    else:
        sys.exit("Error! Please specify your testing file.")

    if os.path.exists(os.path.join(args.save_dir,"model.h5")):
        md = keras.models.load_model(os.path.join(args.save_dir,"model.h5"))
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


