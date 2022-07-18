import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from joblib import Parallel, delayed
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import  SGDClassifier, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import sklearn.discriminant_analysis as sk
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier





def makeModels(classifiers, fselect, path):
    fModels = dict()
    topAcc, bestModel = 0, None
    for f in fselect:
        fModels[f] = list()
        for c in classifiers:
            df = pd.read_csv(path+ c + f + '_finalscore.csv', index_col=0)
            df = dict(df.loc[df['Average Accuracy'].idxmax()])
                
            if c == 'svm':
                if df['Kernel'] == 'linear':
                    model = LinearSVC(C=df['C'], random_state=0, max_iter=100000, dual=False)
                elif df['Kernel'] == 'rbf':
                    model = SVC(kernel = 'rbf', C=df['C'], gamma=df['Gamma'], random_state=0, max_iter=100000)
                elif df['Kernel'] == 'poly':
                    model = SVC(kernel='poly', C=df['C'], gamma=df['Gamma'], degree=df['Degree'], random_state=0, max_iter=10000000)
                #fModels[f].append(('svm', model))
            elif c == 'elasticnet':
                model = SGDClassifier(loss = 'log',alpha= df['Alpha'],penalty = 'l1',random_state=0)
            elif c == 'knn':
                model = KNeighborsClassifier(n_neighbors=df['Neighbors'],n_jobs=-1)
            elif c == 'LDA':
                if df['Solver'] == 'svd':
                    model = sk.LinearDiscriminantAnalysis(solver = 'svd')
                else:
                    model = sk.LinearDiscriminantAnalysis(solver = df['Solver'], shrinkage=df['Shrinkage'])
            elif c == 'logreg':
                model = LogisticRegression(penalty = 'none', max_iter=10000)
            elif c == 'naiveBayes':
                model = BernoulliNB()
            elif c == 'rdforest':
                model = RandomForestClassifier(n_estimators=df['N_Estimators'], max_depth=df['Max_Depth'],  random_state=0, n_jobs=-1)
            elif c == 'xgboost':
                model = XGBClassifier(booster='gbtree',max_depth=df['Max_depth'],learning_rate=0.05,n_estimators = df['N_estimators'], use_label_encoder =False,
                                gamma = df['Gamma'], reg_alpha = df['Alpha'], min_child_weight = df['MinChildWeights'], colsample_bytree = df['byTrees'], reg_lambda = df['Lambda'])
            if (df['Average Accuracy'] > topAcc):
                topAcc = df['Average Accuracy']
                bestModel = model

            fModels[f].append((c, model))
    print('best Model is: ', type(bestModel).__name__)
    return fModels, bestModel
            


def runSKFold(n_seed, splits, data,target):
    '''
    Splitting the data into n_seed of splits folds cross validarion.
    
    Args:
        n_seed: number of cross-validation
        splits: number of folds in each cross validation
    
    Returns:
        data for each of n_seed * splits folds
    '''
    runs = []
    X = np.array(data.drop(target,axis=1))
    y = np.array(data[target])
    result = Parallel(n_jobs=-1)(delayed(execute_skfold)(X,y, splits, seed) for seed in range(n_seed))
    for i in  result:
        for j in i:
            runs.append(j)
    return runs

def execute_skfold(X,y, splits, seed):
    '''
    Splitting the data into splits for each cross-validation.
    
    Args:
        X: The dataset
        y: The target values
        splits: number of folds in each cross validation
        seed: The number of a single cross-validation
    
    Returns:
        data for each of the splits for a single cross-validation
    '''
    skf = StratifiedKFold(n_splits=splits, random_state=seed, shuffle=True)
    result = Parallel(n_jobs=-1)(delayed(execute_split)(X,y, train, test) for train, test in skf.split(X,y))
    return result

def execute_split(X,y, train, test):
    '''
    generate a training and testing sets.
    
    Args:
        X: The dataset
        y: The target values
        train: indices of the training set
        test: indices of the test set
    
    Returns:
        Imputed and fully processed data for each split
    '''
    X = X.copy()
    X = np.array(X)
    X_train, X_test = X[train], X[test]
    y_train, y_test = y[train], y[test]

    arr = [X_train, X_test, y_train, y_test]
    return arr


def baseline(data, target, path):
    '''
    Generate baseline accuracy and f1 score and multivariate and univariate, and save them in a txt file.

    Args:
        data: The dataset after preprocessing
        target: The dependent variable of the dataset
        path: Directory from which the python file is being run
    '''
    f = open(path+'Baseline.txt','w+')
    rate = sum(data[target])/data.shape[0]
    f.write('base line accuracy is '+str( max(rate,1-rate))+'\n')
    f1 = 2*rate/(1+rate)
    f.write('base line f1 value is '+str(f1))
    f.close()