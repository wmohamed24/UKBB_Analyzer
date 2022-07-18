import glob, os
from pyexpat import features
from turtle import towards
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt 
from matplotlib.ticker import PercentFormatter
import shap
import numpy as np
from statistics import mean, median

def shaplyValue(model, X, y, path):

    if not os.path.exists(path+'SHAP/'):
        os.makedirs(path+'SHAP/')
    path = path + 'SHAP/'
    model.fit(X, y)

    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)
    expected_value = explainer.expected_value


    shap.summary_plot(shap_values, X, show = False)
    plt.savefig(path+'SummaryPlot.png', bbox_inches='tight',dpi=300)
    plt.close()

    shap.summary_plot(shap_values, X, plot_type ='bar', show = False)
    plt.savefig(path+'SummaryBarPlot.png', bbox_inches='tight',dpi=300)
    plt.close()

    shap.decision_plot(expected_value, shap_values.values, X, show = False)
    plt.savefig(path+'DecisionPlot.png', bbox_inches='tight',dpi=300)
    plt.close()


    if not os.path.exists(path+'ScatterPlots/'):
        os.makedirs(path+'ScatterPlots/')
    path = path + 'ScatterPlots/'

    for feature in X.columns.tolist():
        shap.plots.scatter(shap_values[:,feature], show = False)
        plt.savefig(path+feature+'ScatterPlot.png', bbox_inches='tight',dpi=300)
        plt.close()
  

def feature_summaryOld(path, features, fselect):
    os.chdir(path+'results/featureSelection')
    f_summary = pd.DataFrame([], columns=['feature']+fselect)
    for f in features:
        f_summary.loc[len(f_summary.index)] = [f]+[0 for x in range(len(fselect))]
    for file in glob.glob("*.txt"):
        if 'fold' in file:
            feature = file.split('_repeat')[0]
        else:
            feature = file.split('trainSelect')[0]
        
        with open(file, 'r') as f:
            index = np.array([float(field) for field in f.read().split()]).astype(int)
        for i in index:
            f_summary.at[i,feature] += 1
    
    f_summary['TOTAL'] = f_summary.sum(axis=1, numeric_only=True)
    f_summary = f_summary.sort_values(by=['TOTAL'], ascending=False)
    f_summary.to_csv(path+'/STATS/FeatureSelectionSummary.csv',index=True)
    return f_summary

def feature_summaryNew(path, features, fselect, n_itr, cutoff):
    if not os.path.exists(path+'/STATS/featureSelection/'):
        os.makedirs(path+'/STATS/featureSelection/')

    if not os.path.exists(path+'results/features/'):
        os.makedirs(path+'results/features/')

    featuresSelected = list()
    f_summaryFinal = pd.DataFrame([], columns=['feature']+fselect)
    os.chdir(path+'results/features')
    for f in features:
        f_summaryFinal.loc[len(f_summaryFinal.index)] = [f]+[0 for x in range(len(fselect))]


    os.chdir(path+'results/featureSelection')
    for fs in fselect:
        open(path+'results/features/'+fs+'Final.txt', 'w').close()
        runs = list(range(1, n_itr+1))
        runs = ['run_'+str(x) for x in runs]
        f_summary = pd.DataFrame([], columns=['feature']+runs)
        for f in features:
            f_summary.loc[len(f_summary.index)] = [f]+[0 for x in range(len(runs))]
        for run in runs:
            file = fs+'_strap_'+str(run.split('_')[1])+'.txt'
            with open(file, 'r') as fileRead:
                index = np.array([float(field) for field in fileRead.read().split()]).astype(int)
            for i in index:
                f_summary.at[i,run] += 1
            
        f_summary['TOTAL'] = f_summary.sum(axis=1, numeric_only=True)
        for x in range(len(f_summary)):
            f_summaryFinal.at[x, fs] = f_summary.at[x, 'TOTAL']
        f_summary = f_summary.sort_values(by=['TOTAL'], ascending=False).reset_index()
        f_summary.to_csv(path+'/STATS/featureSelection/'+fs+'.csv',index=True)

        outcomeIndex = list()
        count = 0
        for x in range(len(f_summary)):
            if (f_summary.at[x, 'TOTAL'])/n_itr >= cutoff or fs == 'AllFeatures':
                outcomeIndex.append(f_summary.at[x, 'index'])
                count +=1
        if not fs == 'AllFeatures':
            featuresSelected.append(count)
        
        toWrite = open(path+'results/features/'+fs+'Final.txt', 'a')
        for x in outcomeIndex:
            toWrite.write(str(x) + ' ')
        toWrite.close()
    
    f_summaryFinal.drop(['AllFeatures'], axis = 1, inplace=True)
    f_summaryFinal['TOTAL'] = f_summaryFinal.sum(axis=1, numeric_only=True)
    f_summaryFinal = f_summaryFinal.sort_values(by=['TOTAL'], ascending=False).reset_index()
    f_summaryFinal.to_csv(path+'/STATS/featureSelection/robust.csv',index=True)

    outcomeIndex = list()
    for x in range(median(featuresSelected)):
        outcomeIndex.append(f_summary.at[x, 'index'])
    
    toWrite = open(path+'results/features/robustFinal.txt', 'a')
    for x in outcomeIndex:
        toWrite.write(str(x) + ' ')
    toWrite.close()
    
    return f_summaryFinal
    
    
        



def create_STATS(path):
    '''
    Gather the max accuracies and percisions of all the classifers and feature selection methods and save them in a csv file.

    Args:
        path: Directory from which the python file is being run
    '''
    os.chdir(path+'/results/classifiers/')
    df = pd.DataFrame(columns=['classFeature','Accuracy', 'F1', 'precision', 'recall', 'balanced_Accuracy'])
    for file in glob.glob("*trainValidation.txt"):
        with open(path+"results/classifiers/" + file,'r') as f:
            lines, ls, confusion = f.readlines(), list(), list()
            for line in lines:
                hold = line.split()
                if hold and hold[0].isdigit():
                    ls.append([int(x) for x in hold])
            for i in range(0, len(ls), 2):
                confusion.append(np.array([ls[i], ls[i+1]]))
            finalConfMatrix = sum(confusion)

            trainVal = open(path+"results/classifiers/" + file, "a")
            trainVal.write(str(finalConfMatrix).replace(' [', '').replace('[', '').replace(']', '') + '\n\n')
            trainVal.close()

            tp = finalConfMatrix[1][1]
            fp = finalConfMatrix[0][1]
            fn = finalConfMatrix[1][0]
            tn = finalConfMatrix[0][0]
            total = sum([tn,fp,fn,tp])

            accuracy = (tp+tn)/total
            precision = 0 if tp + fp == 0 else (tp/(tp+fp))
            recall = 0 if tp+fn == 0 else (tp/(tp+fn))
            f1 = 0 if precision + recall == 0 else 2*((precision*recall)/(precision+recall))
            negative = 0 if (tn+fn) == 0 else (tn/(tn+fn))
            positive = 0 if (tp+fp) == 0 else (tp/(tp+fp))
            balancedAcc = 0.5*(positive+negative)



        df = pd.concat([df, pd.DataFrame({'classFeature':[file],'Accuracy':[accuracy],'F1': [f1], 
        'precision':[precision], 'recall':[recall], 'balanced_Accuracy':[balancedAcc]})], ignore_index=True)

    df.to_csv(path+'/STATS/trainValidation_max_scores_in_summary.csv',index=True)
    
    
    df = pd.DataFrame(columns=['classFeature','Accuracy', 'F1', 'precision', 'recall', 'balanced_Accuracy'])
    for file in glob.glob("*trainTest.txt"):
        with open(path+"results/classifiers/" + file,'r') as f:
            lines, ls, confusion = f.readlines(), list(), list()
            for line in lines:
                hold = line.split()
                if hold and hold[0].isdigit():
                    ls.append([int(x) for x in hold])
            for i in range(0, len(ls), 2):
                confusion.append(np.array([ls[i], ls[i+1]]))
            finalConfMatrix = sum(confusion)

            trainTest = open(path+"results/classifiers/" + file, "a")
            trainTest.write(str(finalConfMatrix).replace(' [', '').replace('[', '').replace(']', '') + '\n\n')
            trainTest.close()

            tp = finalConfMatrix[1][1]
            fp = finalConfMatrix[0][1]
            fn = finalConfMatrix[1][0]
            tn = finalConfMatrix[0][0]
            total = sum([tn,fp,fn,tp])
            

            accuracy = (tp+tn)/total
            precision = 0 if tp + fp == 0 else (tp/(tp+fp))
            recall = 0 if tp+fn == 0 else (tp/(tp+fn))
            f1 = 0 if precision + recall == 0 else 2*((precision*recall)/(precision+recall))
            negative = 0 if (tn+fn) == 0 else (tn/(tn+fn))
            positive = 0 if (tp+fp) == 0 else (tp/(tp+fp))
            balancedAcc = 0.5*(positive+negative)

        df = pd.concat([df, pd.DataFrame({'classFeature':[file],'Accuracy':[accuracy],'F1': [f1], 
        'precision':[precision], 'recall':[recall], 'balanced_Accuracy':[balancedAcc]})], ignore_index=True)

    df.to_csv(path+'/STATS/trainTest_max_scores_in_summary.csv',index=True)


            
        

def heatmap(path, title, trainValid):
    '''
    Create heatmaps for the accuracies and percision scores of all the classifers and feature selection methods.

    Args:
        path: Directory from which the python file is being run
        title: the dependent variable of the dataset
    
    '''
    heatmapspath = path+"STATS/Heatmaps/"
    if not os.path.exists(heatmapspath):
        os.makedirs(heatmapspath)

    if trainValid:
        df = pd.read_csv(path+'/STATS/trainValidation_max_scores_in_summary.csv', index_col=[0])
    else:
        df = pd.read_csv(path+'/STATS/trainTest_max_scores_in_summary.csv', index_col=[0])
    
    df['Accuracy'] = round(df['Accuracy'],3)
    df['F1'] = round(df['F1'],3)
    df['precision'] = round(df['precision'],3)
    df['recall'] = round(df['recall'],3)
    df['balanced_Accuracy'] = round(df['balanced_Accuracy'],3)
    filenames = df["classFeature"]

    featureselection, group, classifier = list(), list(), list()
    for name in filenames:  
        
        num = ""
        for m in name:
            if m.isdigit():
                num += m

        if "elasticnet" in name:
            classifier.append("EN")
        elif "knn" in name:
            classifier.append("KNN")
        elif "naiveBayes" in name:
            classifier.append("NB")
        elif "rdforest" in name:
            classifier.append("RF")
        elif "svm" in name:
            classifier.append("SVM")
        elif "xgboost" in name: 
            classifier.append("XGB")
        elif "logreg" in name: 
            classifier.append("LR")
        elif "LDA" in name:
            classifier.append('LDA')
        elif "NN" in name:
            classifier.append("NN")
        elif "stack" in name:
            classifier.append("stack")
        elif "voting" in name:
            classifier.append("voting")
        elif "semiSuper" in name:
            classifier.append("semiSuper")
            
        
        if "cfs" in name:
            featureselection.append("CFS")
            group.append(1)
        elif "fcbf" in name:
            featureselection.append("FCBF")
            group.append(1)
        elif "mrmr" in name:
            featureselection.append("MRMR-"+num)
            group.append(1)
        elif "infogain" in name:
            featureselection.append("IG-"+num)
            group.append(2)
        elif "fisher" in name:
            featureselection.append("Fish-"+num)
            group.append(2)
        elif "reliefF" in name:
            featureselection.append("ReF-"+num)
            group.append(2)
        elif "jmi" in name:
            featureselection.append("JMI-"+num)
            group.append(1)
        elif "chisquare" in name:
            featureselection.append("CHI2-"+num)
            group.append(2)
        elif "robust" in name:
            featureselection.append("robust")
            group.append(0)
        elif "AllFeatures" in name:
            featureselection.append("All")
            group.append(0)
            
    df["Feature Selection"], df["Classifier"], df["group"] = featureselection, classifier, group
    df = df.sort_values(by = 'group').reset_index().drop('index', axis=1)

    if trainValid:
        df.to_csv(path+'/STATS/trainValidation_max_scores_in_summary.csv',index=True)
    else:
        df.to_csv(path+'/STATS/trainTest_max_scores_in_summary.csv',index=True)

    metrics = ['Accuracy', 'F1', 'precision', 'recall', 'balanced_Accuracy']

    for metric in metrics:

        result = df.pivot(index = ['group', 'Feature Selection'],columns = 'Classifier', values = metric)
        result.reset_index(drop=True, inplace=True, level='group')
        if not result.empty:
            fig,ax = plt.subplots(figsize=(12,7))
            plt.xlabel("Feature Selection", fontsize = 26)
            plt.ylabel("Classifier", fontsize = 26)
            ax.tick_params(labelsize=22)
            
            ax.set_title(title, fontsize = 32)
            res = sns.heatmap(result,fmt ='.1%',cmap ='RdYlGn',annot = True,annot_kws={"size": 16},linewidths=0.30,ax=ax )
            cbar = res.collections[0].colorbar
            cbar.ax.yaxis.set_major_formatter(PercentFormatter(1, 1))
            cbar.ax.tick_params(labelsize=22)
            cbar.ax.locator_params(nbins=2, tight=True)
            cbar.ax.autoscale(enable=False)
            plt.tight_layout()
            if trainValid:
                plt.savefig(heatmapspath + 'trainValid' + metric + '.png')
                plt.close()
            else:
                plt.savefig(heatmapspath + 'trainTest' + metric + '.png')
                plt.close()

        