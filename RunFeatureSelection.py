import featureselection as fselect
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import multiprocessing as mp
from sklearn.utils import resample
import os


def fsOldHelper(myTuple):
    X_train, y_train, train, filename, fold, repeat, fs, path, Valid, X = myTuple
    if Valid:
        toSelect = X_train.values[train,:]
        y_valid = y_train.values[train]
    else:
        toSelect = X_train.values
        y_valid = y_train.values

    runsPath = path+'runs/'
    #toSelect, y_valid = SMOTE(random_state=42).fit_resample(toSelect, y_valid)

    index = fselect.run_feature_selection(fs, toSelect, y_valid)            
    fsFile = open(path+filename, 'w')
    fsFile.write(str(index).replace(',', '').replace('[', '').replace(']', '') + '\n\n')
    fsFile.close()

def fSelectOld(myTuple):
    path, X, y, n_seed, splits, features, columns = myTuple

    print('currently running feature selection')
    input = list()
    for i in range(n_seed):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=i+40)
        cv = StratifiedKFold(n_splits=splits, random_state=i+40, shuffle=True)
        x = 1
        for train, validation in cv.split(X_train, y_train):
            for fs in features:
                filename = fs + '_repeat_' + str(i+1) + '_fold_' + str(x) + '.txt'
                input.append((X_train, y_train, train, filename, x, i+1, fs, path, True, X))
            x +=1
        for fs in features:
            filename = fs + 'trainSelect_repeat_' + str(i+1) + '.txt'
            input.append((X_train, y_train, train, filename, 0, i+1, fs, path, False, X))

    pool = mp.Pool(mp.cpu_count())
    pool.map(fsOldHelper, input)
    pool.close()


def fsNewHelper(myTuple):
    data, target, f, i, path = myTuple

    data = resample(data, replace=True, n_samples=len(data), random_state=i)
    X = data.drop(target, axis = 1).values
    y = data[target].values
    index = fselect.run_feature_selection(f, X, y)
    fname = path + f + '_strap_' + str(i+1) + '.txt'
    fsFile = open(fname, 'w')
    fsFile.write(str(index).replace(',', '').replace('[', '').replace(']', '') + '\n\n')
    fsFile.close()

def fselectNew(myTuple):
    data, target, n_itr, features, path = myTuple
    fPath = path + 'results/featureSelection/'
    if not os.path.exists(fPath):
        os.makedirs(fPath)
    input = list()
    for i in range(n_itr):
        for f in features:
            input.append((data, target, f, i, fPath))

    pool = mp.Pool(mp.cpu_count())
    pool.map(fsNewHelper, input)
    pool.close()
            
    return