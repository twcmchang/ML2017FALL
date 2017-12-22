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
    parser.add_argument('--save_dir', type=str, default='save',
                        help='save directory')
    parser.add_argument('--user_file', type=str, default="data/users.csv",
                        help='input user_file')
    parser.add_argument('--movie_file', type=str, default="data/movies.csv",
                        help='input movie_file')
    args = parser.parse_args()
    test(args)

def test(args):
    if os.path.exists(os.path.join(args.save_dir, "data_loader.pkl")):
        with open(os.path.join(args.save_dir, "data_loader.pkl"), 'rb') as f:
            d = cPickle.load(f)
    else:
        sys.exit("Error! Please put your DataLoader in %s." % args.save_dir)
    
    d = DataLoader()
    if args.user_file is not None:
        d.read_user(user_file=args.user_file)
    if args.movie_file is not None:
        d.read_movie(movie_file=args.movie_file)

    args.n_aux = d.n_aux
    print("Use %d auxiliary meta input" % args.n_aux)

    args.n_usr = d.n_usr
    args.n_mov = d.n_mov

    if args.test_file is not None:
        d.read_test(test_file=args.test_file)
    else:
        sys.exit("Error! Please specify your testing file.")

    args.n_usr = d.n_usr
    args.n_mov = d.n_mov
    args.n_aux = d.n_aux

    if os.path.exists(os.path.join(args.save_dir,"model.h5")):
        md = keras.models.load_model(os.path.join(args.save_dir,"model.h5"))
    else:
        sys.exit("Error! Please specify your model in use.")
    md.summary()

    def myround(x):
        if x<1:
            return(1.0)
        elif x>5:
            return(5.0)
        else:
            return(x)

    y_pred = md.predict([d.test_data[:,0],d.test_data[:,1],d.test_aux[:,:]])
    y_pred = [myround(y_pred[i][0]) for i in range(len(y_pred))]
    output = pd.DataFrame({"TestDataID":np.array(range(len(y_pred)))+1,"Rating":np.array(y_pred)})
    output = output[["TestDataID","Rating"]]
    output.to_csv(args.output, index=False)
    print("Testing completed and saved into %s." % args.output)

if __name__ == '__main__':
    main()


