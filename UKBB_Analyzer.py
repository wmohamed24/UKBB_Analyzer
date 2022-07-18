import pandas as pd
import numpy as np
import os
import kFoldValidationRuns as runs
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample
import StatisticalAnalysis

def main():

    #----------------------------------------------------------------------------------------
    #----------------------------------------------------------------------------------------

    #Data reading and processing

    directory_path = os.path.dirname(os.path.realpath(__file__))+'/' #Directory containing the code and the data
    statsPath = directory_path+'StatisticalAnalysis/' #path for the StatisticalAnalysis Results
    if not os.path.exists(statsPath):
        os.makedirs(statsPath)

    datafile = "SendToWaelOHC" #name of the data file in the data folder
    targetBinary = "GAD7_1" #name of the binarized dependent variable 
    targetContinous = "GAD7" #name of the dependent variable as continues

    #Specify which data file type youa are using
    
    data = pd.read_csv(directory_path+"data/"+datafile+".csv", index_col=0)
    data = data.sample(1000)
    data = resample(data, replace=False, n_samples=int(len(data)/3), random_state=42)


    #data = data.reset_index()
    #data.drop(['PHQ9_1'], axis = 1, inplace = True)
    #dataC = pd.read_csv(directory_path+'data/total_dataset_non_binary.csv', index_col=0)
    #dataC.drop(['PHQ9'], axis = 1, inplace = True)
    #dataC = dataC.reset_index()
    #data['GAD7'] = data['index'].map(dataC.set_index('index')['GAD7'])
    #data = data.set_index('index')
    #data.index.name = None
    #print(undersampleT)
    #undersampleMajor = undersampling(data, targetBinary, 0, 1, 100)
    #undersampleMajor.to_csv(directory_path+'data/undersampleFinal.csv')
    #undersample = pd.read_csv(directory_path+"data/undersampled2.csv", index_col=0)
    #undersample = undersample.loc[undersample[targetBinary] == 0]
    #print(undersample)
    #data = data.reset_index()
    #common_cols = list(set(data.columns) & set(undersample.columns))

    #test = pd.merge(data, undersample, on=common_cols, how = 'inner')
    #print(test[targetBinary].value_counts())
    #print(X[targetBinary].value_counts())

    #cols = data.columns.tolist()
    #cols.remove(targetBinary)

    #for col in cols:
    #    distinct = data[col].dropna().nunique()
    #    if distinct > 10:
    #        cols.remove(col)

    #from sklearn.model_selection import train_test_split
    #stratified_sample, _ = train_test_split(data, test_size=0.999, stratify=data[cols])


    #print(stratified_sample)
    #print(stratified_sample[targetBinary].value_counts())

    
    #----------------------------------------------------------------------------------------
    #----------------------------------------------------------------------------------------
    # 
    # #Classification

    #Keep the classification and feature selection methods that you want
    #classifiers=['logreg', 'xgboost', 'elasticnet', 'naiveBayes', 'LDA', 'rdforest', 'knn', 'svm']
    classifiers = ['xgboost']
    #replace the # with the number of features you want
    fselect=['AllFeatures', 'infogain_50', 'reliefF_50', 'jmi_50', 'mrmr_50', 'chisquare_50', 'fcbf', 'cfs'] 
    #fselect = ['AllFeatures'] 
    #Note that cfs and fcbf find all the significant features so they don't need a number

    n_seed = 2 #Number of validations
    splits = 5 #Number of folds or splits in each validation run
    #cols = dataHotBinary.columns.tolist()
    #cols.remove(targetBinary)
    #unlab = unlab[cols]

    runs.NormalRun(data, directory_path, datafile, targetBinary, classifiers, fselect, n_seed, splits, ensemble=True)

    #----------------------------------------------------------------------------------------
    #----------------------------------------------------------------------------------------

    #Statistical Analysis

    '''
    

    vars = ['General.happiness_1', 'General.happiness_4', 'General.happiness_5', 'General.happiness_3',
    'Townsend.deprivation.index.at.recruitment', 'Ever.addicted.to.any.substance.or.behaviour_1', 'General.happiness_2',
    'rs139459337_2.rs10838524_0', 'sex_1' ,'rs228697_2.rs139459337_2', 'rs1801260_2.rs139459337_2',
    'rs228697_1.rs10462020_2', 'rs228697_2.rs10462020_2', 'rs10462020_2.rs139459337_2', 'rs228697_2.rs10462020_1',
    'rs139459337_2.rs2287161_0', 'rs10462020_2.rs1801260_2', 'Age.at.recruitment', 'rs17031614_2.rs139459337_2',
    'rs10462020_2.rs10462023_2', 'rs10462020_2.rs10462023_1', 'rs10462023_2.rs139459337_2', 'rs10462020_2.rs17031614_1',
    'rs10462020_1.rs139459337_2', 'rs139459337_0.rs2287161_1', 'rs10462023_1', 'rs10462023_1.rs139459337_0', 'rs10838524_1',
    'rs139459337_0.rs10838524_1', targetContinous]

    data = data[vars]
    
    
    cat = pd.read_csv(directory_path + 'data/furtherTesting.csv', index_col=0)
    data = data.reset_index()
    cat = cat.reset_index()
    datacols = data.columns.tolist()
    nonmod = list()
    for col in cat.columns:
        nonmod.append(col)
        if col not in datacols:
            data[col] = data['index'].map(cat.set_index('index')[col])

    asData = data[nonmod]
    asData = asData.set_index('index')
    asData.index.name = None
    asData.drop(['PHQ9'], axis = 1, inplace=True)

    continuous = list()
    for col in asData.columns:
        distinct = asData[col].dropna().nunique()
        if distinct > 10:
            continuous.append(col)
    continuous.remove('GAD7')
    asData.drop(continuous, axis = 1, inplace=True)
    sn = ['rs228697',
       'rs10462020', 'rs10462023', 'rs1801260', 'rs17031614', 'rs139459337',
       'rs10838524', 'rs2287161']
    templist = list(sn)
    for s in sn:
        templist.remove(s)
        for s2 in templist:
            #asData[str(s) + '_' + str(s2)] = asData[[str(s), str(s2)]].agg('_'.join, axis=1) 
            asData[str(s) + '_' + str(s2)] = asData[str(s)].astype(str) + '_' + asData[str(s2)].astype(str)

    asData.columns = asData.columns.str.replace(' ', '_')
    #print(asData)
    #print(asData['rs139459337_rs2287161'].value_counts())

    asData = asData[asData['rs139459337_rs2287161'] != 'nan_nan']

    cols = asData.columns.tolist()
    cols.remove('sex')
    cols.remove('GAD7')
    for col in cols:
        StatisticalAnalysis.ANOVA(data=asData, path=directory_path, dep='GAD7', indep=['sex',col], oneWay = False)



    #print(asData)
    
    #assAnalyis = StatisticalAnalysis.Association_Analysis(data = asData, path = directory_path, vars = asData.columns.tolist(), oneTarget = True, target = 'GAD7_1')
    '''
    '''
    data.drop([targetContinous], axis = 1, inplace=True)
    continuous = list()
    for col in data.columns:
        distinct = data[col].dropna().nunique()
        if distinct > 10:
            continuous.append(col)
    data.drop(continuous, axis = 1, inplace=True)
    for col in data.columns:
        if '.' in col:
            data.drop(col, axis = 1, inplace = True)
    print(data.columns)
    print(data)
    '''
    '''
    vars = pd.read_csv(directory_path+'data/FeatureSelectionSummary.csv')['feature'].tolist()[:40]

    snps = ['rs228697_1','rs228697_2','rs10462020_1','rs10462020_2','rs10462023_1','rs10462023_2',
    'rs1801260_1','rs1801260_2','rs17031614_1','rs17031614_2','rs139459337_1','rs139459337_2','rs10838524_1',
    'rs10838524_2','rs2287161_1','rs2287161_2']

    for snp in snps:
        if snp not in vars:
            vars.append(snp)

    vars.append(targetBinary)
    data = data[vars]
    data.columns = data.columns.str.replace(' ', '.')
    
    '''
    '''
    data.columns = data.columns.str.replace(' ', '_')

    for col in data.columns:
        if '.' in col:
            data.drop([col], axis = 1, inplace=True)
    continuous = list()
    categorical = list()
    for col in data.columns:
        distinct = data[col].dropna().nunique()
        if distinct > 10:
            continuous.append(col)  
        elif distinct > 2:
            categorical.append(col)
        
    if targetContinous in continuous:
        continuous.remove(targetContinous)
    data.drop(continuous, axis = 1, inplace=True)

    for col in data.columns:
        if not col == 'GAD7' and not col == 'sex_1':
            StatisticalAnalysis.ANOVA(data=data, path=directory_path, dep='GAD7', indep=[col], oneWay = True)
            break
    '''

    #print(cat)
    #print (continuous)
    #print(data['GAD7'])
    #Uncomment the staistical test desired and pass the suitable parameters
    #MultiReg = StatisticalAnalysis.Multivariate_Reg(data = data, path = directory_path, target=targetContinous, continuous = continuous, stepwise = True, categorical = categorical)
    #oddsRatios = StatisticalAnalysis.Odds_Ratios(data = data, path = directory_path, target=targetBinary, continuous=continuous, stepwise=False)
    #assAnalyis = StatisticalAnalysis.Association_Analysis(data = data, path = directory_path, vars = data.drop([targetBinary]+continuous, axis = 1))
    #assRuleLearning = StatisticalAnalysis.Association_Rule_Learning(data=data, path = directory_path, rhs = 'GAD7_1')

main()