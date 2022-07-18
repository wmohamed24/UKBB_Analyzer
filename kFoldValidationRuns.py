import os
import RunClassifiers as runC
import ClassFeatureStats as st
import multiprocessing as mp
import numpy as np
from imblearn.over_sampling import SMOTE
import RunFeatureSelection as runFs


def NormalRun(data, directory_path, datafile, target, classifiers, fselect, n_seed, splits, unlabData = None, ensemble = False):
    
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
        type: Type of analysis performed
        n_seed: Number of validations
        splits: Number of folds or splits in each validation run  
        continuous: list of the names of continuous variables
        columns_org: names of columns before dummification 
    '''
    
    target_path = directory_path+'NormalDataAnalysis/'+datafile+"_"+target+"/"

    results_path = target_path+"results/"
    STATS_path = target_path+"STATS/"
    Features_path = results_path+"featureSelection/"
    classifiersPath = results_path+"classifiers/"
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    if not os.path.exists(STATS_path):
        os.makedirs(STATS_path)
    if not os.path.exists(Features_path):
        os.makedirs(Features_path)
    if not os.path.exists(classifiersPath):
        os.makedirs(classifiersPath)

    f = open(STATS_path+'Baseline.txt','w+')
    rate = sum(data[target])/data.shape[0]
    f.write('base line accuracy is '+str( max(rate,1-rate))+'\n')
    f1 = 2*rate/(1+rate)
    f.write('base line f1 value is '+str(f1))
    f.close()

    data = data.copy(deep = True)
    X = data.drop(target, axis = 1)
    y = data[target]

    mp.set_start_method('fork')
    pool = mp.Pool(int(mp.cpu_count()))
    
    classifiersInput = list()
    stackInput = list()

    

    #runFs.fselectNew((data, target, 100, fselect, target_path))
    #st.feature_summaryNew(target_path, X.columns, fselect, 100, 0.7)
    fselect.append('robust')

    for i in classifiersInput:
       runC.classify(i)
    #rsr.Stack((target_path, classifiers, fselect, X, y, n_seed, splits))
    
    st.create_STATS(target_path)
    st.heatmap(target_path, target, True)
    st.heatmap(target_path, target, False)
    
    
