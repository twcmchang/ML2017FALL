import numpy as np
import pandas as pd
import random
import time
import util
import pickle

ADD_FEATURE = False
NORMALIZATION = False
MODEL_NO = 2

duration = 9
recorder = pd.DataFrame()

print("data loaded with duration = %d" % duration)
train_all = util.load_train("train.csv", duration = duration)

# Just remove all records with any feature < 0
train_data = train_all[(train_all >= 0).all(1)]

# Shuffle before split features and outcome
train_data = train_data.reindex(np.random.permutation(train_data.index))
train_data.index = range(train_data.shape[0])

# Split input and output
y = pd.Series(train_data['y'])
del train_data['y']

if ADD_FEATURE:
    print("creating features...")

    add_list = list(train_data.columns)
    train_data = util.create_ft(train_data, add_list)
    config['add_list'] = add_list

if NORMALIZATION:
    print("normalizing...")

    train_data = (train_data - train_data.mean()) / train_data.std()
    config['norm_mean'] = train_data.mean()
    config['norm_std'] = train_data.std()

if MODEL_NO == 2:
    print("change to using model (2)")
    extCol = [ colname for colname in train_data.columns if 'PM2.5' in colname]
    train_data = train_data[extCol]

## add constant 1
train_data.insert(loc = 0, column = 'intercept', value = 1)

test_file   = 'test.csv'
out_file    = 'out_'+str(MODEL_NO)+'_'+str(duration)+'_normal.csv'
config_pkl  = 'config_'+str(MODEL_NO)+'_'+str(duration)+'_normal.pickle'
coef_pkl    = 'coef_'+str(MODEL_NO)+'_'+str(duration)+'_normal.pickle'

config = {}
config['duration'] = duration

iteration = 5000
lr = 1e-6
lb = 1e-6
optimizer = 'GD'
decay = 0

print("setting: \n- iteration = %d \n- learning rate = %7f \n- lambda = %7f \n- optimizer = %s \n- decay = %d"
        % (iteration, lr, lb, optimizer, decay))

loss_history = list()
coef = np.zeros(train_data.shape[1])
stime = time.time()
for ite in range(iteration):
    if decay>0 and ite%decay==0 and ite>0:
        lr = lr/2
        print("current lr=%.8f" %lr)
    if optimizer == 'GD':
        coef, rmse = util.update_GD(train_data, y, coef, lr, lb=lb)
    elif optimizer == 'Adagrad':
        coef, rmse = util.update_Adagrad(train_data, y, coef, lr, lb=lb)
    loss_history.append(rmse)
    if ite%100==0:
        print("epoch %d, time: %2f, loss: %2f" % (ite, time.time()-stime, rmse))
        stime = time.time()

recorder.insert(0,column=str(duration)+'_'+str(lb),value=loss_history)

pickle.dump(config, open(config_pkl,'wb'))
pickle.dump(coef, open(coef_pkl,'wb'))
util.test(test_file, out_file, config_pkl, coef_pkl, MODEL_NO)

