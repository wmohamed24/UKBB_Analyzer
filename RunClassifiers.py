import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import KFold, GridSearchCV, StratifiedKFold
from imblearn import FunctionSampler
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, make_scorer, balanced_accuracy_score
from skopt import BayesSearchCV
from sklearn.linear_model import  SGDClassifier, LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
import sklearn.discriminant_analysis as sk
from sklearn.svm import SVC
from xgboost import XGBClassifier
from skopt.space import Real, Integer, Categorical
import multiprocessing as mp
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import ignore_warnings
from imblearn.combine import SMOTETomek
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import cross_val_score
import re
import os
from ast import literal_eval

class MyClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, estimator, method, path, splits, run, n_estimators = 100, max_depth = 10, 
    penalty = 'none', alpha = 0.01, n_neighbors = 5, solver = 'svd', shrinkage = 0,
    kernel = 'linear', C = 1, gamma = 0.1, degree = 3, eta = 0.05, scale_pos_weight = 1,
    min_samples_split = 2, min_samples_leaf = 1, l1_ratio = 0.15, leaf_size = 30,
    subsample = 3, colsample_bytree = 5, min_child_weight = 3, class_weight = 0.5):

        self.idx, self.method, self.estimator, self.splits = 0, method, estimator, splits
        self.max_depth, self.n_estimators = max_depth, n_estimators
        self.penalty, self.alpha, self.n_neighbors = penalty, alpha, n_neighbors
        self.solver, self.shrinkage = solver, shrinkage
        self.C, self.gamma, self.degree, self.kernel = C, gamma, degree, kernel
        self.eta, self.repeat, self.path = eta, -1, path
        self.scale_pos_weight, self.min_child_weight = scale_pos_weight, min_child_weight
        self.min_samples_split, self.min_samples_leaf = min_samples_split, min_samples_leaf
        self.l1_ratio, self.leaf_size, self.class_weight = l1_ratio, leaf_size, class_weight
        self.subsample, self.colsample_bytree = subsample, colsample_bytree,
        self.run = run


        if self.estimator == 'rdforest':
            self.classify = RandomForestClassifier(n_estimators=self.n_estimators, max_depth= self.max_depth,
            min_samples_split=self.min_samples_split, min_samples_leaf = self.min_samples_leaf)
            self.hyperParam = ['class_weight', 'max_depth', 'min_samples_leaf', 'min_samples_split', 'n_estimators']
            
        elif self.estimator == 'logreg':
            self.classify = LogisticRegression(penalty = self.penalty, max_iter=10000)
            self.hyperParam = ['class_weight', 'penalty']
        
        elif self.estimator == 'elasticnet':
            self.classify = SGDClassifier(loss = 'log_loss', alpha= self.alpha, penalty = 'l1', 
            random_state=0, l1_ratio = self.l1_ratio)
            self.hyperParam = ['alpha', 'class_weight', 'l1_ratio']
        
        elif self.estimator == 'naiveBayes':
            self.classify = BernoulliNB(alpha=self.alpha)
            self.hyperParam = ['alpha']

        elif self.estimator == 'knn':
            self.classify = KNeighborsClassifier(n_neighbors=self.n_neighbors, leaf_size = self.leaf_size)
            self.hyperParam = ['leaf_size', 'n_neighbors']

        elif self.estimator == 'LDA':
            self.classify = sk.LinearDiscriminantAnalysis(solver = self.solver, shrinkage=self.shrinkage)
            self.hyperParam = ['shrinkage', 'solver']
        
        elif self.estimator == 'svm':
            if self.kernel == 'linear':
                self.classify = SVC(kernel = 'linear', C=self.C, random_state=0, max_iter=100000)
            elif self.kernel == 'rbf':
                self.classify = SVC(kernel = 'rbf', C=self.C, gamma=self.gamma, random_state=0, max_iter=100000)
            elif self.kernel == 'poly':
                self.classify = SVC(kernel='poly', C=self.C, gamma=self.gamma, degree=self.degree, random_state=0, max_iter=10000000)
            self.hyperParam = ['C', 'class_weight', 'degree', 'gamma', 'kernel']

        elif self.estimator == 'xgboost':
            self.classify = XGBClassifier(booster='gbtree',max_depth=self.max_depth,eta=self.eta,
            min_child_weight = self.min_child_weight, subsample = self.subsample, colsample_bytree = self.colsample_bytree,
            n_estimators = self.n_estimators, use_label_encoder =False, scale_pos_weight = self.scale_pos_weight)
            self.hyperParam = ['colsample_bytree', 'eta', 'max_depth', 'min_child_weight', 
            'n_estimators', 'scale_pos_weight', 'subsample']
            

    def set_params(self, **params):
        for x in params:
            setattr(self, x, params[x])
            
        if 'class_weight' in params:
            params['class_weight'] = {0:params['class_weight'], 1:1-params['class_weight']}

        self.classify.set_params(**params)


    @ignore_warnings(category=ConvergenceWarning)
    def fit(self, X, y):
        #X,y = SMOTE(random_state=42).fit_resample(X,y)


        fname = self.path + 'results/features/' +self.method+ 'Final.txt'
        with open(fname, 'r') as f:
            index = np.array([float(field) for field in f.read().split()]).astype(int)

        self.idx = index
        X = X[:,self.idx]
        return self.classify.fit(X,y)

    def predict(self, X, y=None):
        X = X[:,self.idx]
        return self.classify.predict(X)

    def score(self, X, y, sample_weight=None):
        yp = self.predict(X)
        f1 = f1_score(y, yp, sample_weight=sample_weight)
        acc = accuracy_score(y, yp, sample_weight=sample_weight)
        balanced = balanced_accuracy_score(y, yp)
        conf = confusion_matrix(y, yp)

        tempList = list()
        for par in self.hyperParam:
            tempList.append(getattr(self,par))

        file = open(self.path+'results/hyperParamsRuns/'+self.estimator+'_'+self.method+'_run_'+str(self.run)+'.txt', 'a')    
        file.write(str(tempList)+':'+str(conf[0][0])+' '+str(conf[0][1])+' '+str(conf[1][0])+' '+str(conf[1][1])+'\n')
        file.close()

        return f1

    def makeTuple(self):
        tempList = list()
        for par in self.hyperParam:
            tempList.append(getattr(self,par))
        return tuple(tempList)


def resample(X, y):
    X, y = SMOTETomek(random_state=42).fit_resample(X,y)
    print(X.shape)
    return X,y
    X, y = SMOTE().fit_resample(X, y)
    return X,y


def getParameters(estimator):
    if estimator =='rdforest':
        return {'n_estimators': Integer(200,2000), 'max_depth': Integer(5,100),
        'min_samples_split':Integer(2,10), 'min_samples_leaf':Integer(1,5),
        'class_weight': Real(0.01, 0.99, prior = 'log-uniform')}, 40
    elif estimator == 'logreg':
        return {'penalty': ['none'], 'class_weight': Real(0.01, 0.99, prior = 'log-uniform')}, 20
    elif estimator == 'elasticnet':
        return {'alpha': Real(1e-5, 100, prior = 'log-uniform'),
        'l1_ratio': Real(0.01, 0.99, prior = 'log-uniform'),
        'class_weight': Real(0.01, 0.99, prior = 'log-uniform')}, 20
    elif estimator == 'naiveBayes':
        return {'alpha': [1]}, 1
    elif estimator == 'knn':
        return {'n_neighbors': Integer(2,10), 'leaf_size': Integer(10,100)}, 20
    elif estimator == 'LDA':
        return {'shrinkage' : Real(0.01, 0.99, prior = 'log-uniform'), 'solver': ['lsqr']}, 20
    elif estimator == 'svm':
        return {'degree': Real(2, 10), 'gamma':  Real(1e-3, 1e3, prior = 'log-uniform'),
        'C': Real(1e-3, 1e3, prior = 'log-uniform'), 'kernel': ['linear', 'rbf', 'poly'],
        'class_weight': Real(0.01, 0.99, prior = 'log-uniform')}, 40
    elif estimator == 'xgboost':
        return {'eta':Real(0.005, 0.5, prior = 'log-uniform'), 'max_depth': Integer(3,15),
        'subsample' : Real(0.01, 0.99, prior = 'log-uniform'), 'scale_pos_weight' : Integer(1,100),
        'colsample_bytree' : Real(0.01, 0.99, prior = 'log-uniform'), 
        'n_estimators':Integer(50, 500),  'min_child_weight' : Integer(5,10)}, 60
	

def classify(myTuple):
    target_path, X, y, n_seed, splits, c, fs, columns = myTuple
    cpath = target_path+'results/classifiers/'
    fPath = target_path+'results/features/'
    rPath = target_path+'results/hyperParamsRuns/'
    if not os.path.exists(rPath):
        os.makedirs(rPath)

    open(cpath+c+fs+"trainValidation.txt", 'w').close()
    open(cpath+c+fs+"trainTest.txt", 'w').close()

    for i in range(n_seed):
        open(rPath+c+"_"+fs+"_run_"+str(i+1)+".txt", "w").close() 
        print('currently running: ' + c + ' with ' + fs + ' repeat ' + str(i+1))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=i+40)
        _pipeline = make_pipeline(MyClassifier(estimator=c, method=fs, splits = splits, path = target_path, run = i+1))
        cv = StratifiedKFold(n_splits=splits, random_state=i+40, shuffle= True)

        parms, itr = getParameters(c)
        parms = {'myclassifier__' + key: parms[key] for key in parms}

        grid_imba = BayesSearchCV(_pipeline, search_spaces=parms, cv=cv, n_iter=itr, refit = False, n_jobs=-1, n_points=5)
        grid_imba.fit(X_train.values, y_train.values)
        best = grid_imba.best_params_

        bestParam = dict()
        for key in best.keys():
            bestParam[key.split('__')[1]] = best[key]

        if 'class_weight' in bestParam:
            bestParam['class_weight'] = {0:bestParam['class_weight'], 1:1-bestParam['class_weight']}

        confusion = dict()
        with open(rPath+c+"_"+fs+"_run_"+str(i+1)+".txt", "r") as f:
            for line in f.readlines():
                params, matrix = line.split(':')[0], line.split(':')[1].split(' ')
                params = literal_eval(params)
                matrix = [int(x) for x in matrix]
                matrix = np.array([[matrix[0], matrix[1]],[matrix[2], matrix[3]]])

                if tuple(params) in confusion:
                    confusion[tuple(params)].append(matrix)
                else:
                    confusion[tuple(params)] = [matrix]

        confMatrix = sum(confusion[tuple(best.values())])
        confusion = dict()
        
        trainVal = open(cpath+c+fs+"trainValidation.txt", "a")
        trainVal.write(str(grid_imba.best_params_)+'\n')
        trainVal.write(str(confMatrix).replace(' [', '').replace('[', '').replace(']', '') + '\n\n')
        trainVal.close()

        
        if c == 'rdforest':
            Tester = RandomForestClassifier(**bestParam)
        elif c == 'logreg':
            Tester = LogisticRegression(**bestParam, max_iter=10000)
        elif c == 'elasticnet':
            Tester = SGDClassifier(**bestParam, loss = 'log_loss', penalty = 'l1')
        elif c == 'naiveBayes':
            Tester = BernoulliNB(**bestParam)
        elif c == 'knn':
            Tester = KNeighborsClassifier(**bestParam)
        elif c == 'LDA':
            Tester = sk.LinearDiscriminantAnalysis(**bestParam)
        elif c == 'svm':
            Tester = SVC(**bestParam, random_state=0, max_iter=10000000)
        elif c == 'xgboost':
            Tester = XGBClassifier(**bestParam, use_label_encoder = False)

        X_t, y_t = X_train.values, y_train.values
        #X_t, y_t = SMOTE(random_state=42).fit_resample(X_t, y_t)
        
        fname = fPath+fs+'Final.txt'
        with open(fname, 'r') as f:
            index = np.array([float(field) for field in f.read().split()]).astype(int)
        X_t = X_t[:, index]
        X_test = X_test.values[:, index]

        Tester.fit(X_t, y_t)
        yp = Tester.predict(X_test)

        trainTest = open(cpath+c+fs+"trainTest.txt", "a")
        confMatrix = confusion_matrix(y_test,yp)
        trainTest.write(str(grid_imba.best_params_)+'\n')
        trainTest.write(str(confMatrix).replace(' [', '').replace('[', '').replace(']', '') + '\n\n')
        trainTest.close()
        
def getModel(key, params):
        if key == 'xgboost':
            return XGBClassifier(**params, use_label_encoder = False)
        elif key == 'elasticnet':
            return SGDClassifier(**params, loss = 'log_loss', penalty = 'l1')
        elif key == 'LDA':
            return sk.LinearDiscriminantAnalysis(**params)
        elif key == 'logreg':
            return LogisticRegression(**params, max_iter=10000)
        elif key == 'naiveBayes':
            return BernoulliNB(**params)
        elif key == 'rdforest':
            return RandomForestClassifier(**params)
        elif key == 'svm':
            return SVC(**params, random_state=0, max_iter=10000000)
        elif key == 'knn':
            return KNeighborsClassifier(**params)

def StackRun(myTuple):
    X_train, y_train, X_test, y_test, filename, model, fselect = myTuple
    print(filename)

    with open(fselect, 'r') as f:
        index = np.array([float(field) for field in f.read().split()]).astype(int)
    X_train = X_train[:,index]
    model.fit(X_train,y_train)
    yp = model.predict(X_test[:, index])

    conf = confusion_matrix(y_test, yp)
    confusionMatrix = open(filename, "a")
    confusionMatrix.write(str(conf).replace(' [', '').replace('[', '').replace(']', '') + '\n\n')
    confusionMatrix.close()



def Stack(myTuple):

    path, classifiers, features, X, y, n_seed, splits = myTuple
    stackInput = list()
    for fs in features:
        for i in range(n_seed):
            X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.1, random_state=i+40)
            cv = StratifiedKFold(n_splits=splits, random_state=i+40, shuffle= True)
            level0 = list()
            level1 = list()

            for c in classifiers:
                with open(path+'results/classifiers/'+c+fs+'trainValidation.txt') as f:
                    data = f.readlines()[i*4].split('OrderedDict(')[1]
                data = '[' + data[1:-3] + ']'
                data = [tuple(x.split(',')) for x in re.findall("\((.*?)\)", data)]
                params = dict()
                for num in range(len(data)):
                    a = data[num][0].replace("'","").split('__')[1]
                    a = a.replace(" ", "")
                    if "'" in data[num][1]:
                        b = data[num][1].replace("'","")
                        b = b.replace(" ", "")
                    elif '.' in data[num][1] or 'e' in data[num][1]:

                        b = float(data[num][1])
                    else:
                        b = int(float(data[num][1]))
                    if a == 'class_weight':
                        b = {0:b, 1:1-b}
                    params[a] = b
                level0.append((c, getModel(c, params)))
                if c == 'logreg':
                    level1 = getModel(c, params)
            
            if not level1:
                level1 = LogisticRegression()

            stack = StackingClassifier(estimators=level0, final_estimator=level1)
            x = 1
            filename = path + 'results/classifiers/' + 'stacking' + fs + 'trainValidation.txt'
            open(filename, 'w').close()
            for train, validation in cv.split(X_train, y_train):
                feature = path + 'results/featureSelection/' + fs + '_repeat_' + str(i+1) + '_fold_' + str(x) + '.txt'
                stackInput.append((X_train[train,:], y_train[train], X_train[validation,:], y_train[validation], filename, stack, feature))
                x += 1

            filename = path + 'results/classifiers/' + 'stacking' + fs + 'trainTest.txt'
            open(filename, 'w').close()
            feature = path + 'results/featureSelection/' + fs + 'trainSelect' + '_repeat_' + str(i+1) + '.txt'
            stackInput.append((X_train, y_train, X_test, y_test, filename, stack, feature))

    pool = mp.Pool(mp.cpu_count())
    pool.map(StackRun, stackInput)
    pool.close()
            
            
        