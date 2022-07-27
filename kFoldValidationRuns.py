import os
import RunClassifiers as runC
import ClassFeatureStats as st
import multiprocessing as mp
import numpy as np
from imblearn.over_sampling import SMOTE
import RunFeatureSelection as runFs


def NormalRun(data, directory_path, datafile, target, classifiers, fselect, n_seed, splits, 
doC, doF, cluster, fselectRepeat, cutoff, robustFeatures):
    
    '''
    Create all the directories to store all the intermediate and final results. 
    Impute and prepare the data for running.
    Splits the data and run the feature selection and classification on runs.

    Args:
        data : The DataFrame that contains data that hasn't been preprocessed.
        directory_path: Directory from which the python file is being run
        datafile: name of the datafile inside the data folder
        target: the dependent variable of the dataset
        classifiers: Classification methods used
        fselect: Feature selection methods used
        n_seed: Number of validations
        splits: Number of folds or splits in each validation run  
        doC: boolean indicating whether to run Classification
        doF: boolean indicating whether to run featureSelection 
        cluster: set to True if classification will be run on a cluster
        fselectRepeat: num. of repeations in feature selection bootstrapping
        cutoff: cutoff for selecting features from different methods
        robustFeatures: num. of features to be selected as robust

    '''
    
    target_path = directory_path+'NormalDataAnalysis/'+datafile+"_"+target+"/"

    results_path = target_path+"results/"
    STATS_path = target_path+"STATS/"
    Features_path = results_path+"featureSelection/"
    classifiersPath = results_path+"classifiers/"
    hyperParmPath = results_path+"hyperParamsRuns/"
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    if not os.path.exists(STATS_path):
        os.makedirs(STATS_path)
    if not os.path.exists(Features_path):
        os.makedirs(Features_path)
    if not os.path.exists(classifiersPath):
        os.makedirs(classifiersPath)
    if not os.path.exists(hyperParmPath):
        os.makedirs(hyperParmPath)

    ## calculate baseline values for accuracy and f1-score
    f = open(STATS_path+'Baseline.txt','w+')
    rate = sum(data[target])/data.shape[0]
    f.write('base line accuracy is '+str( max(rate,1-rate))+'\n')
    f1 = 2*rate/(1+rate)
    f.write('base line f1 value is '+str(f1))
    f.close()

    ## break data into predictive variables and outcome variable
    data = data.copy(deep = True)
    X = data.drop(target, axis = 1)
    y = data[target]

    ## run feature selection
    if doF:
        runFs.fselectNew((data, target, fselectRepeat, fselect, target_path))
        st.feature_summaryNew(target_path, X.columns, fselect, fselectRepeat, cutoff, robustFeatures)
    fselect.append('robust')
    
    ## run classification on the cluster
    classifiersInput = list()
    for fs in fselect:
        for c in classifiers:
            classifiersInput.append((target_path, X, y, n_seed, splits, c, fs, data.columns))

    if doC and cluster:
        pList = list()
        for i in range(len(classifiersInput)):
            pid = os.fork()
            if pid:
                pList.append(pid)
            else:
                runC.classify(classifiersInput[i])
                os._exit(0)

        for i, child in enumerate(pList):
            os.waitpid(child, 0)                        

    ## run classification on personal computer
    elif doC:
        for arg in classifiersInput:
            runC.classify(arg)

    ## create hearmaps for classification
    if doC:
        st.create_STATS(target_path)
        st.heatmap(target_path, target, True)
        st.heatmap(target_path, target, False)
    
    
