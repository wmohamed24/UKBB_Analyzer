import pandas as pd
import numpy as np
import os
import kFoldValidationRuns as runs
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample
from StatisticalAnalysis import Stats
import multiprocessing as mp



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

def distillTotal(dir,features_target,features):
    file=open('NormalDataAnalysis/'+dir+'/results/features/robustFinal.txt','r')
    line=[int(i) for i in file.readline().split() if i!='None']
    index=np.where(features==features_target)[0][0]
    line=np.array([i+1 if i>=index else i for i in line])
    #print(index)
    #print(line)
    line=features[line]
    line=np.append(line,features_target)
    #print(len(line))
    #print(line)
    new_features=distill(line)
    #print(len(new_features))
    #print(np.array(new_features))
    return np.array(new_features)
    
    
def importantFeatures(dir,data,target):
    file=open('NormalDataAnalysis/'+dir+'/results/features/robustFinal.txt','r')
    line=[int(i) for i in file.readline().split() if i!='None']
    index=np.where(data.columns==target)[0][0]
    line=np.array([i+1 if i>=index else i for i in line])
    line=np.array(data.columns[line])
    line=np.append(line,target)
    return line
    #to_drop=np.array(data.columns[np.setdiff1d(np.array([i for i in range(len(data.columns))]),line)])
    #print(np.setdiff1d(data.columns,to_drop))
    #to_drop=np.delete(to_drop,np.where(np.array(to_drop)==target)[0])
    #data.drop(labels=to_drop,axis=1,inplace=True)

def main():

    #----------------------------------------------------------------------------------------
    #----------------------------------------------------------------------------------------

    #Data reading and processing

    directory_path = os.path.dirname(os.path.realpath(__file__))+'/' #Directory containing the code and the data
    statsPath = directory_path+'StatisticalAnalysis/' #path for the StatisticalAnalysis Results
    if not os.path.exists(statsPath):
        os.makedirs(statsPath)

    datafile = "run1OHC" #name of the data file in the data folder
    target = "GAD7_1" #name of the binarized dependent variable 


    #Specify which data file type youa are using
    
    data = pd.read_csv(directory_path+"Data/"+datafile+".csv", index_col=0)

    #----------------------------------------------------------------------------------------
    #----------------------------------------------------------------------------------------
    # 
    # #Classification
    mp.set_start_method('fork')
    #Keep the classification and feature selection methods that you want
    classifiers=['xgboost', 'LDA', 'rdforest', 'logreg']#, 'svm']
    #replace the # with the number of features you want
    fselect=['AllFeatures', 'infogain_50', 'reliefF_50', 'jmi_50', 'mrmr_50', 'chisquare_50']     #, 'fisher_50', 'fcbf', 'cfs'] 
    #fselect = ['AllFeatures', 'mrmr_50', 'chisquare_50']
    #Note that cfs and fcbf find all the significant features so they don't need a number

    n_seed = 5 #Number of validations
    splits = 10 #Number of folds or splits in each validation run

    #runs.NormalRun(data, directory_path, datafile, target, classifiers, fselect, n_seed, splits, doC=True,doF=False,cluster=True,fselectRepeat=0,cutoff=0.7,robustFeatures=25)

    #----------------------------------------------------------------------------------------
    #----------------------------------------------------------------------------------------
    
    #Statistical Analysis
    
    #cols = ['Townsend deprivation index at recruitment', 'Ever addicted to any substance or behaviour_1',
    #'rs10462020_2.rs17031614_2', 'rs228697_2.rs10462020_2', 'rs228697_2.rs10462020_1', 'rs228697_1.rs10462020_2', 'GAD7_1']
    
    #features_datafile='run1OHC'
    #features = np.array(pd.read_csv(directory_path+"Data/"+features_datafile+".csv", index_col=0).columns)
    #cols=distillTotal('run1OHC_GAD7_1','GAD7_1',features)
    cols=importantFeatures('run1OHC_GAD7_1',data,target)
    

    data = data[cols]
    data.columns = data.columns.str.replace(' ', '_')
    data.columns = data.columns.str.replace('.', '_')

    #for col in cols:
        #print(data[col].value_counts())
    
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
    
    data.drop(continuous, axis = 1, inplace=True)
    #print(data.columns)
    
    #Uncomment the staistical test desired and pass the suitable parameters
    sa=Stats(directory_path,statsPath)
    #sa.ANOVA(data = data, dep = target, indep='rs10462020_rs1801260', oneWay=True, followUp=False)
    
    #data.drop(labels=continuous,axis=1,inplace=True)
    # for var in data.columns:
    #     if var != 'GAD7' and var != 'sex':

    #         sa.ANOVA(data = data, dep = target, indep=['rs139459337_rs10838524'], oneWay=True, followUp=True)
    #         break
    
    temp = np.array(data['GAD7_1'].values)
    temp -=1
    temp = np.absolute(temp)
    data['GAD7_1'] = temp
    print(data['GAD7_1'].value_counts())


    #MultiReg = sa.Multivariate_Reg(data = data, target=target, continuous = continuous, stepwise = True, categorical = categorical)
    #oddsRatios = sa.Odds_Ratios(data = data, target=target, continuous=continuous, stepwise=False, categorical=categorical)
    #assAnalyis = sa.Association_Analysis(data = data, vars = data.drop(continuous, axis = 1), oneTarget=True, target=target)
    #assRuleLearning = sa.Association_Rule_Learning(data=data.drop(labels=continuous,axis=1,inplace=False), rhs = 'GAD7_1',max_items=4,min_confidence=0.01,min_support=0.0001)
    #sa.Mediation_Analysis(data=data, dep = 'GAD7', mediator='Townsend_deprivation_index_at_recruitment', indep = 'rs10838524_2_rs2287161_1', path = statsPath, continuous=continuous)
    #sa.Mendelian_Ranomization(data = data, dep = 'GAD7_1', indep = 'Chronotype_1', inst='rs10838524_2_rs2287161_1', path = statsPath)
    
main()