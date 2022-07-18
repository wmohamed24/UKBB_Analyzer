from math import gamma
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import  SGDClassifier, LogisticRegression
import sklearn.discriminant_analysis as sk
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import ignore_warnings
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.semi_supervised import LabelPropagation
from bayes_opt import BayesianOptimization
#from keras.models import Sequential
#from keras.layers import Flatten
#from keras.layers import Dropout
#from keras.layers import Dense
#from tensorflow.keras.optimizers import Adam


'''
def make_model(hiddenLayerOne=784, hiddenLayerTwo=256,
	dropout=0.2, learnRate=0.01, n = 0):

    model = Sequential()
    model.add(Dense(hiddenLayerOne, input_dim=n, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(hiddenLayerTwo, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(1, activation="sigmoid"))


    model.compile(
		optimizer=Adam(learning_rate=learnRate),
		loss="binary_crossentropy",
		metrics=["accuracy"])
    return model


def NeuralNetworks(X_train,X_test,y_train,y_test, n):

    
    Create Neural Netowrk models on X_train and y_train with different parameters.
    Runs the models on X_test and compare the results with y_test.
    
    Args:
        X_train: A Numpy array containing the dataset for training
        X_test: A Numpy array containing the dataset for testing
        y_train: A Numpy array consisting of the target values for training
        y_test: A Numpy array consisting of the target values for testing

    Returns:
        A DataFrame with all the paramaters used and confusion matrices of each model
    
    print(n, '\n\n')
    df = pd.DataFrame(columns=['batchSize', 'epochs', 'hiddenLayerOne', 'hiddenLayerTwo', 'dropout', 'learnRate', 'Confusion Matrix'])
    rows = []

    hiddenLayerOne = [256, 512, 784]
    hiddenLayerTwo = [128, 256, 512]
    learnRate = [1e-2, 1e-3, 1e-4]
    dropout = [0.3, 0.4, 0.5]
    batchSize = [4, 8, 16, 32]
    epochs = [10, 20, 30, 40]

    for batch in batchSize:
        for epo in epochs:
            for layer1 in hiddenLayerOne:
                for layer2 in hiddenLayerTwo:
                    for drop in dropout:
                        for rate in learnRate:
                            model = make_model(layer1, layer2, drop, rate, n)
    
                            model.fit(X_train, y_train, batch, epo)
                            predicted_labels = (model.predict(X_test) > 0.5).astype(int)
                            tn, fp, fn, tp = confusion_matrix(y_test, predicted_labels, labels=[0,1]).ravel()
                            convert_matrix = [tn,fp,fn,tp]
                            rows.append([batch, epo, layer1, layer2, drop, rate, convert_matrix])

    for i in range(len(rows)):
        df = pd.concat([df, pd.DataFrame({'batchSize':[rows[i][0]],'epochs':[rows[i][1]], 'hiddenLayerOne': [rows[i][2]], 
        'hiddenLayerTwo': [rows[i][3]], 'dropout': [rows[i][4]], 'learnRate': [rows[i][5]],'Confusion Matrix':[rows[i][6]]})], ignore_index=True)

    return df

'''

@ignore_warnings(category=ConvergenceWarning)
def StackClassifier(X_train,X_test,y_train,y_test, models):
    
    '''
    Creates an ensemble classifier based on the models provided using stacking.
    
    Args:
        X_train: A Numpy array containing the dataset for training
        X_test: A Numpy array containing the dataset for testing
        y_train: A Numpy array consisting of the target values for training
        y_test: A Numpy array consisting of the target values for testing
        models: the models to be used for creating the ensemble

    Returns:
        A DataFrame with all the paramaters used and confusion matrices of each model
    '''

    model = StackingClassifier(estimators=models, cv=10)
    df = pd.DataFrame(columns=['Confusion Matrix'])

    model.fit(X_train, y_train)
    predicted_labels = model.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, predicted_labels, labels=[0,1]).ravel()
    convert_matrix_b = [tn,fp,fn,tp]
    df['Confusion Matrix'] = [convert_matrix_b]
    return df

@ignore_warnings(category=ConvergenceWarning)
def VoteClassifier(X_train,X_test,y_train,y_test, models):

    '''
    Creates an ensemble classifier based on the models provided using voting.
    
    Args:
        X_train: A Numpy array containing the dataset for training
        X_test: A Numpy array containing the dataset for testing
        y_train: A Numpy array consisting of the target values for training
        y_test: A Numpy array consisting of the target values for testing
        models: the models to be used for creating the ensemble

    Returns:
        A DataFrame with all the paramaters used and confusion matrices of each model
    '''

    model = VotingClassifier(estimators=models, voting='soft')
    df = pd.DataFrame(columns=['Confusion Matrix'])

    model.fit(X_train, y_train)
    predicted_labels = model.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, predicted_labels, labels=[0,1]).ravel()
    convert_matrix_b = [tn,fp,fn,tp]
    df['Confusion Matrix'] = [convert_matrix_b]
    return df



def LinearDiscriminntAnalyzer(X_train,X_test,y_train,y_test):

    '''
    Create Linear Discriminnt Analyzer models on X_train and y_train with different parameters.
    Runs the models on X_test and compare the results with y_test.
    
    Args:
        X_train: A Numpy array containing the dataset for training
        X_test: A Numpy array containing the dataset for testing
        y_train: A Numpy array consisting of the target values for training
        y_test: A Numpy array consisting of the target values for testing

    Returns:
        A DataFrame with all the paramaters used and confusion matrices of each model
    '''
    
    solvers = ['svd', 'lsqr']
    shrinkage = np.arange(0, 1, 0.01)
    df = pd.DataFrame(columns=['Shrinkage', 'Solver','Confusion Matrix'])
    rows = []

    for s in solvers:
        if s =='svd':
            LDA = sk.LinearDiscriminantAnalysis(solver = s)
            LDA.fit(X_train,y_train)
            predicted_labels = LDA.predict(X_test)

            tn, fp, fn, tp = confusion_matrix(y_test, predicted_labels, labels=[0,1]).ravel()
            convert_matrix = [tn,fp,fn,tp]
            rows.append(['', s, convert_matrix])
        else:
            for r in shrinkage:
                LDA = sk.LinearDiscriminantAnalysis(solver = s, shrinkage=r)
                LDA.fit(X_train,y_train)
                predicted_labels = LDA.predict(X_test)

                tn, fp, fn, tp = confusion_matrix(y_test, predicted_labels, labels=[0,1]).ravel()
                convert_matrix = [tn,fp,fn,tp]
                rows.append([r, s, convert_matrix])

    for i in range(len(rows)):
        df = pd.concat([df, pd.DataFrame({'Shrinkage':[rows[i][0]],'Solver':[rows[i][1]],'Confusion Matrix':[rows[i][2]]})], ignore_index=True)

    return df

def elasticnet(X_train,X_test,y_train,y_test):
    '''
    Create multiple Elasticnet classifier models on X_train and y_train with different parameters.
    Runs the models on X_test and compare the results with y_test.
    
    Args:
        X_train: A Numpy array containing the dataset for training
        X_test: A Numpy array containing the dataset for testing
        y_train: A Numpy array consisting of the target values for training
        y_test: A Numpy array consisting of the target values for testing

    Returns:
        A DataFrame with all the paramaters used and confusion matrices of each model
    '''
    df = pd.DataFrame(columns=['Alpha','Confusion Matrix'])
    rows = []
    alphas= [0.0001,0.0005,  0.0008, 0.001,0.002,0.003,0.004,0.005, 0.01]
    for al in alphas:
        regr = SGDClassifier(loss = 'log',alpha= al,penalty = 'l1',random_state=0)
        model = regr.fit(X_train, y_train)
        predicted_labels = model.predict(X_test)
        
        tn, fp, fn, tp = confusion_matrix(y_test, predicted_labels, labels=[0,1]).ravel()
        convert_matrix = [tn,fp,fn,tp]
        rows.append([al,convert_matrix])
    for i in range(len(rows)):
        df.loc[len(df.index)] = rows[i] 


    return df

def logistic_regression(X_train, X_test, y_train, y_test):
    '''
    Creates multiple Logitic Regression classifier models on X_train and y_train with different parameters.
    Runs the models on X_test and compare the results with y_test.
    
    Args:
        X_train: A Numpy array containing the dataset for training
        X_test: A Numpy array containing the dataset for testing
        y_train: A Numpy array consisting of the target values for training
        y_test: A Numpy array consisting of the target values for testing

    Returns:
        A DataFrame with all the paramaters used and confusion matrices of each model
    '''
    model = LogisticRegression(penalty = 'none', max_iter=10000)
    model.fit(X_train, y_train)
    predicted_labels = model.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, predicted_labels, labels=[0,1]).ravel()
    convert_matrix = [tn,fp,fn,tp]
    df = pd.DataFrame()
    df['Confusion Matrix'] = [convert_matrix]
    return df

    

    

def KNN(X_train,X_test,y_train,y_test):
    '''
    Creates multiple KNearestNeighbors classifier models on X_train and y_train with different parameters.
    Runs the models on X_test and compare the results with y_test.
    
    Args:
        X_train: A Numpy array containing the dataset for training
        X_test: A Numpy array containing the dataset for testing
        y_train: A Numpy array consisting of the target values for training
        y_test: A Numpy array consisting of the target values for testing

    Returns:
        A DataFrame with all the paramaters used and confusion matrices of each model
    '''
    neighbors = [5,10,12,14,16,20]
    df = pd.DataFrame(columns=['Neighbors','Confusion Matrix'])
    rows = []

    for n in neighbors:
        knn = KNeighborsClassifier(n_neighbors=n,n_jobs=-1)
        knn.fit(X_train,y_train)
        predicted_labels = knn.predict(X_test)

        tn, fp, fn, tp = confusion_matrix(y_test, predicted_labels, labels=[0,1]).ravel()
        convert_matrix = [tn,fp,fn,tp]
        rows.append([n, convert_matrix])

    for i in range(len(rows)):
        df.loc[len(df.index)] = rows[i] 

    return df

@ignore_warnings(category=ConvergenceWarning)
def SVM(X_train,X_test,y_train,y_test):
    '''
    Creates Support Vector Machine classifier models on X_train and y_train with different parameters.
    Runs the models on X_test and compare the results with y_test.
    
    Args:
        X_train: A Numpy array containing the dataset for training
        X_test: A Numpy array containing the dataset for testing
        y_train: A Numpy array consisting of the target values for training
        y_test: A Numpy array consisting of the target values for testing

    Returns:
        A DataFrame with all the paramaters used and confusion matrices of each model
    '''

    df = pd.DataFrame(columns=['Kernel','C','Gamma','Degree','Confusion Matrix'])
    rows = []

    Cs = [1e-3, 1e-2, 1e-1, 1e0, 1e1]
    gammas = [1e-2, 1e-1]
    degrees = [2, 3]

    for c in Cs:
        linear = LinearSVC(C=c, random_state=0, max_iter=100000, dual=False)
        linear.fit(X_train, y_train)
        predicted_labels = linear.predict(X_test)
        tn, fp, fn, tp = confusion_matrix(y_test, predicted_labels, labels=[0,1]).ravel()
        convert_matrix = [tn,fp,fn,tp]
        rows.append(['linear', c, '', '', convert_matrix])

        for gamma in gammas:
            rbf = SVC(kernel = 'rbf', C=c, gamma=gamma, random_state=0, max_iter=100000)
            rbf.fit(X_train, y_train)
            predicted_labels = rbf.predict(X_test)
            tn, fp, fn, tp = confusion_matrix(y_test, predicted_labels, labels=[0,1]).ravel()
            convert_matrix = [tn,fp,fn,tp]
            rows.append(['rbf', c, gamma, '', convert_matrix])

            for degree in degrees:
                poly = SVC(kernel='poly', C=c, gamma=gamma, degree=degree, random_state=0, max_iter=10000000)
                poly.fit(X_train,y_train)
                predicted_labels = poly.predict(X_test)
                tn, fp, fn, tp = confusion_matrix(y_test, predicted_labels, labels=[0,1]).ravel()
                convert_matrix = [tn,fp,fn,tp]
                rows.append(['poly', c, gamma, degree, convert_matrix])

    for i in range(len(rows)):
        
        df.loc[len(df.index)] = rows[i]
        
    return df



def rdforest(X_train,X_test,y_train,y_test):
    '''
    Creates multiple Random Forest classifier models on X_train and y_train with different parameters.
    Runs the models on X_test and compare the results with y_test.
    
    Args:
        X_train: A Numpy array containing the dataset for training
        X_test: A Numpy array containing the dataset for testing
        y_train: A Numpy array consisting of the target values for training
        y_test: A Numpy array consisting of the target values for testing

    Returns:
        A DataFrame with all the paramaters used and confusion matrices of each model
    '''
    df = pd.DataFrame(columns=['N_Estimators','Max_Depth','Confusion Matrix'])
    rows = []

    estimators = [200,300,400,500]
    max_depths = [5,7,10]

    for estimator in estimators:
        for max_d in max_depths:
            rdf = RandomForestClassifier(n_estimators=estimator, max_depth=max_d,  random_state=0, n_jobs=-1)
            rdf.fit(X_train, y_train)
            predicted_labels = rdf.predict(X_test)
            tn, fp, fn, tp = confusion_matrix(y_test, predicted_labels, labels=[0,1]).ravel()
            convert_matrix = [tn,fp,fn,tp]
            rows.append([estimator, max_d, convert_matrix])

    for i in range(len(rows)):
        df.loc[len(df.index)] = rows[i] 
    return df


def xgboost(X_train,X_test,y_train,y_test):
    '''
    Creates multiple XgBoost classifier models on X_train and y_train with different parameters.
    Runs the models on X_test and compare the results with y_test.
    
    Args:
        X_train: A Numpy array containing the dataset for training
        X_test: A Numpy array containing the dataset for testing
        y_train: A Numpy array consisting of the target values for training
        y_test: A Numpy array consisting of the target values for testing

    Returns:
        A DataFrame with all the paramaters used and confusion matrices of each model
    '''
    
    df = pd.DataFrame(columns=['Max_depth','N_estimators', 'Gamma', 'Alpha', 'byTrees', 'MinChildWeights', 'Lambda','Confusion Matrix'])
    rows = []

    gammas = [0] #[0, 1, 3, 5, 9] #[0]
    reg_alphas = [0] #np.linspace(0, 180, 4, dtype=int) #[0]
    reg_lambdas = [1] #[0, 0.3, 0.7, 1] #[1]
    colsample_bytrees = [1] #[0.5, 0.8, 1] #[1]
    min_child_weights = [1] #[0, 2, 4, 6, 8]

    rate = 0.05
    max_depth = [3,4,5,6,7]
    n_estimators= np.linspace(50, 450, 4, dtype=int)

    for depth in max_depth:
        for estimators in n_estimators:
            for g in gammas:
                for alp in reg_alphas:
                    for colsample in colsample_bytrees:
                        for minChild in min_child_weights:
                            for lam in reg_lambdas:
                                xgb = XGBClassifier(booster='gbtree',max_depth=depth,learning_rate=rate,n_estimators = estimators, use_label_encoder =False,
                                gamma = g, reg_alpha = alp, min_child_weight = minChild, colsample_bytree = colsample, reg_lambda = lam)
                                xgb.fit(X_train, y_train)
                                predicted_labels = xgb.predict(X_test)
                                tn, fp, fn, tp = confusion_matrix(y_test, predicted_labels, labels=[0,1]).ravel()
                                convert_matrix = [tn,fp,fn,tp]
                                rows.append([depth,estimators, g, alp, colsample, minChild, lam,convert_matrix])

    for i in range(len(rows)):
        df.loc[len(df.index)] = rows[i] 

    return df


def naive_bayes(X_train,X_test,y_train,y_test):
    '''
    Creates multiple Naive Bayes classifier models on X_train and y_train with different parameters.
    Runs the models on X_test and compare the results with y_test.
    
    Args:
        X_train: A Numpy array containing the dataset for training
        X_test: A Numpy array containing the dataset for testing
        y_train: A Numpy array consisting of the target values for training
        y_test: A Numpy array consisting of the target values for testing

    Returns:
        A DataFrame with all the paramaters used and confusion matrices of each model
    '''
    df = pd.DataFrame(columns=['Confusion Matrix'])

    bnb = BernoulliNB()
    bnb.fit(X_train, y_train)
    predicted_labels = bnb.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, predicted_labels, labels=[0,1]).ravel()
    convert_matrix_b = [tn,fp,fn,tp]
    df['Confusion Matrix'] = [convert_matrix_b]
    return df


@ignore_warnings(category=ConvergenceWarning)
def SemiSupervisedLearning(X_train,X_test,y_train,y_test, unlabData, bestModel):

    '''
    Creates LabelPropagation classifier that will be used to classify unlabeled data 
    passed which will be then used as a training data along with the passed training data.
    The final predictions are made using the classifier passed as bestMoel
    
    Args:
        X_train: A Numpy array containing the dataset for training
        X_test: A Numpy array containing the dataset for testing
        y_train: A Numpy array consisting of the target values for training
        y_test: A Numpy array consisting of the target values for testing
        unlabData: A Numpy array consisting of the ulabeled data
        bestModel: the best Classifier found after testing all the classifiers

    Returns:
        A DataFrame with all the paramaters used and confusion matrices of each model
    '''

    df = pd.DataFrame(columns=['Confusion Matrix'])
    
    X_train_mixed = np.concatenate((X_train, unlabData))
    nolabel = [-1 for _ in range(len(unlabData))]
    y_train_mixed = np.concatenate((y_train, nolabel))
    model = LabelPropagation(max_iter=1000)

    model.fit(X_train_mixed, y_train_mixed)
    tran_labels = model.transduction_
    bestModel.fit(X_train_mixed, tran_labels)
    predicted_labels = bestModel.predict(X_test)
    
    tn, fp, fn, tp = confusion_matrix(y_test, predicted_labels, labels=[0,1]).ravel()
    convert_matrix_b = [tn,fp,fn,tp]
    df['Confusion Matrix'] = [convert_matrix_b]
    return df

def classify(estimator, X_train, X_test, y_train, y_test, models, unlabData, bestModel):
    '''
    Runs the specific Classification method.

    Args:
        X_train: A Numpy array containing the dataset for training
        X_test: A Numpy array containing the dataset for testing
        y_train: A Numpy array consisting of the target values for training
        y_test: A Numpy array consisting of the target values for testing

    Returns:
        A DataFrame with all the paramaters used and confusion matrices of each model of the specified Classifier
    '''
    if estimator == 'svm':
        return SVM(X_train, X_test, y_train, y_test)
    elif estimator == 'naiveBayes':
        return naive_bayes(X_train, X_test, y_train, y_test)
    elif estimator == 'rdforest':
        return rdforest(X_train, X_test, y_train, y_test)
    elif estimator == 'knn':
        return KNN(X_train, X_test, y_train, y_test)
    elif estimator == 'elasticnet':
        return elasticnet(X_train, X_test, y_train, y_test)
    elif estimator =='xgboost':
        return xgboost(X_train, X_test, y_train, y_test)
    elif estimator =='logreg':
        return logistic_regression(X_train, X_test, y_train, y_test)
    elif estimator == 'LDA':
        return LinearDiscriminntAnalyzer(X_train, X_test, y_train, y_test)
    elif estimator == 'stack':
        return StackClassifier(X_train, X_test, y_train, y_test, models)
    elif estimator == 'voting':
        return VoteClassifier(X_train, X_test, y_train, y_test, models)
    elif estimator == 'semiSuper':
        return SemiSupervisedLearning(X_train, X_test, y_train, y_test, unlabData, bestModel)
    #elif estimator == 'NN':
        #return NeuralNetworks(X_train, X_test, y_train, y_test, n)