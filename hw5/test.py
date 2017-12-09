import numpy as np
import pandas as pd
import os
import sys
import keras
import argparse
from six.moves import cPickle
from model import MF
from utils import DataLoader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_file', type=str, default="data/test.csv",
                        help='testing filename')
    parser.add_argument('--output', type=str, default=None,
                        help='output filename')
    parser.add_argument('--save_dir', type=str, default='save/',
                        help='save directory')
    args = parser.parse_args()
    test(args)

def test(args):
    d = DataLoader()
    if args.test_file is not None:
        d.read_test(test_file=args.test_file)
    else:
        sys.exit("Error! Please specify your testing file.")

    if os.path.exists(os.path.join(args.save_dir,"model.h5")):
        md = keras.models.load_model(os.path.join(args.save_dir,"model.h5"))
    else:
        sys.exit("Error! Please specify your model in use.")
    md.summary()

    y_pred = md.predict([d.test_data[:,0],d.test_data[:,1]])
    y_pred = [y_pred[i][0] for i in range(len(y_pred))]
    output = pd.DataFrame({"TestDataID":np.array(range(len(y_pred)))+1,"Rating":np.array(y_pred)})
    output = output[["TestDataID","Rating"]]
    output.to_csv(args.output, index=False)
    print("Testing completed and saved into %s." % args.output)

if __name__ == '__main__':
    main()


