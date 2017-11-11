## (1) Import packages
import numpy as np
import pandas as pd
import math
import pickle

def create_ft(X, list_a):
    for key in list_a:
        b = np.square(X[key])
        X["square_"+key] = b.fillna(b.min()-1)
    return X

def update_Adagrad(X, y, coef, lr, lb = None):
    y_hat = np.dot(X, coef)
    loss = y_hat - y
    mse = np.sum(loss**2)/len(y)
    rmse = math.sqrt(mse)
    grad = np.dot(X.transpose(),loss)/len(y)
    s_grad = grad**2
    ada = np.sqrt(s_grad)
    if lb==None:
        coef = coef - lr*grad/ada
    else:
        coef = coef - lr*grad/ada + (lb/len(y))*coef
    return coef, rmse

def update_GD(X, y, coef, lr, lb = None):
    y_hat = np.dot(X, coef)
    loss = y_hat - y
    mse = np.sum(loss**2)/len(y)
    rmse = math.sqrt(mse)
    grad = np.dot(X.transpose(),loss)/len(y)
    if lb == None:
        coef = coef - lr*grad
    else:
        coef = coef - lr*grad + (lb/len(y))*coef
    return coef, rmse

def load_train(filename, duration=9):
    ## (1) Read training data
    train = pd.read_csv(filename, encoding = 'big5')
    # Change column names
    train.columns = ['date','station','item','0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23']
    # Delete column 'station'
    train = train.drop(train.columns[[1]],1)

    ## (3) Reshape the training data to be one value per row
    train_melt = pd.melt(train, id_vars=['date', 'item'], value_vars=[str(s) for s in list(range(24))])
    train_melt['date'] = pd.to_datetime(train_melt['date'])
    train_melt['variable'] = pd.to_numeric(train_melt['variable'])
    # Convert all values into float and set 'NR' in 'RAINFALL' to be 0.0
    train_melt['value'] = pd.to_numeric(train_melt['value'], errors = 'coerce')
    train_melt.fillna(0, inplace = True)
    # Sort all values according to 'date' and 'variable' (hour in 0~23)
    train_melt.sort_values(by = ['date', 'variable'], ascending = [1, 1], inplace = True)
    train_melt.index = range(train_melt.shape[0])

    ## (4) Extract all duration-hour window
    # Since there are 18 items, start from 0 and then 18, ... and so on
    for i in range(0, train_melt.shape[0]-1, 18):
        if i + (duration*19) > train_melt.shape[0]-1:
            break
        elif (train_melt.iloc[i+(duration*19),0].replace(hour = int(train_melt.iloc[i+(duration*19),2])) - train_melt.iloc[i,0].replace(hour = int(train_melt.iloc[i,2]))) / np.timedelta64(1, 'h') > duration:
            continue
        tmp = train_melt.iloc[np.append(np.arange(i,i+duration*18),i+(duration*18+9)), 3]
        if i == 0:
            train_all = tmp
        else:
            train_all = pd.concat([train_all.reset_index(drop=True), tmp.reset_index(drop=True)], axis=1)
    train_all = train_all.T
    train_all.index = range(train_all.shape[0])

    ## (5) Set feature names (and let outcome to be 'y')
    featureNames = list()
    names = ['h'+str(i) for i in np.arange(duration)]
    for hr in names:
        for var in train.ix[0:17, 1]:
            featureNames.append(var + '_' + hr)
    featureNames.append('y')
    train_all.columns = featureNames
    return train_all

def load_test(test_file, duration):
    test = pd.read_csv(test_file, encoding='big5', header = None)
    # Change column names
    test.columns = ['id','item','h1','h2','h3','h4','h5','h6','h7','h8','h9']
    # Melt (one value per row)
    test_melt = pd.melt(test, id_vars=['id', 'item'], value_vars=['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7', 'h8', 'h9'])
    test_melt['id'] = pd.to_numeric([s[3:] for s in test_melt['id']]).astype(int)
    test_melt['value'] = pd.to_numeric(test_melt['value'], errors = 'coerce')
    test_melt.fillna(0, inplace = True)
    test_melt.sort_values(by = ['id', 'variable'], ascending = [1, 1], inplace = True)
    test_melt.index = range(test_melt.shape[0])
    # Expand (one record per row)
    # Start from 0 and then 18, ... and so on
    for i in range(0, test_melt.shape[0]-1, 162):
        tmp = test_melt.iloc[range(i,i+18*duration), 3]
        if i == 0:
            test_all = tmp
        else:
            test_all = pd.concat([test_all.reset_index(drop=True), tmp.reset_index(drop=True)], axis=1)
    test_all = test_all.T
    test_all.index = range(test_all.shape[0])
    # Change column names: h1 to h(duration)
    featureNames = list()
    names = ['h'+str(i) for i in np.arange(duration)]
    for hr in names:
        for var in test.ix[0:17, 1]:
            featureNames.append(var + '_' + hr)
    test_all.columns = featureNames
    return test_all

def test(test_file, out_file, config_pkl, coef_pkl, MODEL_NO):
    print("reading configuration file...")
    config = pd.read_pickle( open( config_pkl, "rb" ) )

    if 'duration' in config.keys():
        duration = config['duration']
        print("reading test file...")
        test_data = load_test(test_file, duration = duration)
    else:
        print("Error: not specify observation duration.")
        return False
    
    if 'add_list' in config.keys():
        print("creating features...")
        add_list = config['add_list']
        test_data = create_ft(test_data, add_list)

    if ('norm_mean' in config.keys()) and ('norm_std' in config.keys()):
        print("normalizing...")
        norm_mean = config['norm_mean']
        norm_std = config['norm_std']
        test_data = (test_data - norm_mean) / norm_std

    if MODEL_NO==2:
        extCol = [colname for colname in test_data.columns if 'PM2.5' in colname]
        test_data = test_data[extCol]

    test_data.insert(loc = 0, column = 'intercept', value = 1)

    print("loading trained coefficients...")

    coef = pickle.load( open( coef_pkl, "rb" ) )

    ## (10) Predict
    result = test_data.dot(coef)
    result.index = ['id_' + str(s) for s in range(240)]
    result.to_csv(out_file, index_label = ['id'], header = ['value'])

    print("completed and written in %s" % out_file)
    return True

