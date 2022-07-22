import pandas as pd
import numpy as np
import os
import kFoldValidationRuns as runs
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample
import StatisticalAnalysis



def distill(features):
    new_features=[]
    for i in range(len(features)):
        feature=features[i]
        if '.' in feature:
            working=feature.split('.')
            new_feature='.'.join([working[0].split('_')[0],working[1].split('_')[0]])
        elif '_' in feature:
            new_feature=feature.split('_')[0]
        else:
            new_feature=feature

        if new_feature not in new_features:
            new_features.append(new_feature)
    return new_features

def main():

    #----------------------------------------------------------------------------------------
    #----------------------------------------------------------------------------------------

    #Data reading and processing

    directory_path = os.path.dirname(os.path.realpath(__file__))+'/' #Directory containing the code and the data
    statsPath = directory_path+'StatisticalAnalysis/' #path for the StatisticalAnalysis Results
    if not os.path.exists(statsPath):
        os.makedirs(statsPath)

    datafile = "sendToWaelOHC" #name of the data file in the data folder
    target = "GAD7_1" #name of the binarized dependent variable 


    #Specify which data file type youa are using
    
    data = pd.read_csv(directory_path+"data/"+datafile+".csv", index_col=0)

    #----------------------------------------------------------------------------------------
    #----------------------------------------------------------------------------------------
    # 
    # #Classification
    #Keep the classification and feature selection methods that you want
    classifiers=['xgboost', 'LDA', 'rdforest', 'logreg']#, 'svm']
    #replace the # with the number of features you want
    #fselect=['AllFeatures', 'infogain_50', 'reliefF_50', 'jmi_50', 'mrmr_50', 'chisquare_50', 'fisher_50', 'fcbf', 'cfs'] 
    fselect = ['AllFeatures', 'mrmr_50', 'chisquare_50']
    #Note that cfs and fcbf find all the significant features so they don't need a number

    n_seed = 5 #Number of validations
    splits = 10 #Number of folds or splits in each validation run

    runs.NormalRun(data, directory_path, datafile, target, classifiers, fselect, n_seed, splits, ensemble=True)

    #----------------------------------------------------------------------------------------
    #----------------------------------------------------------------------------------------

    #Statistical Analysis

    

    #cols = distill(cols)

    #data = data[cols]
    data.columns = data.columns.str.replace(' ', '_')
    data.columns = data.columns.str.replace('.', '_')
    
    continuous = list()
    categorical = list()
    for col in data.columns:
        distinct = data[col].dropna().nunique()
        if distinct > 10:
            continuous.append(col)  
        elif distinct > 2:
            categorical.append(col)
    
    if target in continuous:
        continuous.remove(target)

    #data.drop(continuous, axis = 1, inplace=True)
    #print(data.columns)
    #vars = data.columns.tolist()
    #vars.remove('GAD7')
    #StatisticalAnalysis.Association_Analysis(data = data, path = statsPath, vars = vars, oneTarget=True, target='GAD7')

    #for var in data.columns:
    #    if not var == 'sex' and not var == 'GAD7':
    #StatisticalAnalysis.ANOVA(data = data, path = statsPath, dep = target, indep=['sex'], oneWay=True)
    #Uncomment the staistical test desired and pass the suitable parameters

    #MultiReg = StatisticalAnalysis.Multivariate_Reg(data = data, path = statsPath, target=target, continuous = continuous, stepwise = True, categorical = categorical)
    #oddsRatios = StatisticalAnalysis.Odds_Ratios(data = data, path = statsPath, target=target, continuous=continuous, stepwise=True, categorical=categorical)
    #assAnalyis = StatisticalAnalysis.Association_Analysis(data = data, path = directory_path, vars = data.drop([targetBinary]+continuous, axis = 1))
    #assRuleLearning = StatisticalAnalysis.Association_Rule_Learning(data=data, path = statsPath, rhs = 'GAD7_1')
    #StatisticalAnalysis.Mediation_Analysis(data=data, dep = 'GAD7', mediator='Townsend_deprivation_index_at_recruitment', indep = 'rs10838524_2_rs2287161_1', path = statsPath, continuous=continuous)
    #StatisticalAnalysis.Mendelian_Ranomization(data = data, dep = 'GAD7_1', indep = 'Chronotype_1', inst='rs10838524_2_rs2287161_1', path = statsPath)
main()