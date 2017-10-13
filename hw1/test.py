import sys
import pickle
import util # self-defined functions

test_file 	= sys.argv[1]
out_file 	= sys.argv[2]
config_pkl	= "save/config.pickle"
coef_pkl 	= "save/coef.pickle"
MODEL_NO 	= 1

if util.test(test_file, out_file, config_pkl, coef_pkl, MODEL_NO):
	print("Testing completed.")
else:
	print("Error! please contact d05921027@ntu.edu.tw.")
