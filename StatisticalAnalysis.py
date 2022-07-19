import numpy as np
import statsmodels.formula.api as smf
from linearmodels.iv import IV2SLS
import pandas as pd
import statsmodels.api as sm
from bioinfokit.analys import stat
from mne.stats import fdr_correction
import researchpy as rp
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import os
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod import families
from rpy2.robjects.packages import importr
from rpy2.robjects import r, pandas2ri, numpy2ri
numpy2ri.activate()
from rpy2.robjects.conversion import localconverter
import rpy2.robjects as ro
import itertools




plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

    
def Association_Rule_Learning(data, path, min_support = 0.0045, min_confidence = 0.2, min_items = 2, max_items = 5, rhs = 'none'):
    ''' 
    Do Association Rules mining for the items within the passed dataframe. Write all the found 
    association rules that meet the specified conditions and save the produced graphs
    in the passed parameters to AssociationRules.txt in Apriori Folder in the passed path.
    
    Args:
        data: DataFrame containing the items of interest
        path: Folder path to which the data will be saved
        min_support: Minimum value of support for the rule to be considered
        min_confidence: Minimum value of confidence for the rule to be considered
        min_items: Minimum number of item in the rules including both sides
        max_items: Maximum number of item in the rules including both sides
        rhs: If it's desired for one item to be always on the rhs
    Returns:
        A dataframe of all the association rules found
    '''
    data = data.copy()
    for col in data.columns:
        if len(data[col].unique()) > 2:
            data.drop([col], axis = 1, inplace = True)
    
 
    if not os.path.exists(path+'Apriori'):
        os.makedirs(path+'Apriori')
    currPath = path + 'Apriori/'

    data.to_csv(currPath + 'AprioriData.csv')
    args = currPath + ' ' + str(min_support) + ' '  + str(min_confidence) + ' ' + str(max_items) + ' ' + str(min_items) + ' ' + str(rhs)
    os.system('Rscript ' + path + 'Association_Rules.R ' + args)
    os.remove(currPath + 'AprioriData.csv') 
    AssRules = pd.read_csv(currPath + 'apriori.csv')

    Association = open(currPath + 'AssociationRules.txt', 'w')
    Association.write(AssRules.to_string(index=False))
    Association.close()
    os.remove(currPath + 'apriori.csv') 
    
    return AssRules


def Mediation_Analysis(data, dep, mediator, indep, path, continuous = list()):
    '''
    Do Mediation Analysis between the passed dependent & indepent variables 
    and the mediation variable(s) passed in the passed data frame. 
    Write the results to Mediation_Analysis.txt in the passed path
    
    Args:
        data: DataFrame containing the items of interest
        dep: The dependent varaible in the analysis
        mediator: The mediation variable in the analysis
        indep: The independent variable(s) in the analysis - can be a list
        path: Folder path to which the data will be saved
        continuous: list containing continuous variables

    Returns:
        A dictionary mapping each of the independent variables to a dataframe of the mediation analysis
    '''
    data = data.copy(deep = True)

    if not os.path.exists(path + 'MediationAnalysis/'):
        os.makedirs(path +'MediationAnalysis/')
    currPath = path + 'MediationAnalysis/'

    if type(indep) == str:
        t = list(); t.append(indep)
        indep = t

    for var in indep:
        filePath = currPath+'MedAnalysis-'+str(var) + '-' + str(mediator) + '-' + str(dep) + '.txt'

        l1 = importr('mediation')
        formulaMake = r['as.formula']
        mediate, lm, glm, summary, capture = r['mediate'], r['lm'], r['glm'], r['summary'], r['capture.output']

        MediationFormula = formulaMake(mediator + ' ~ ' + var)
        OutcomeFormula = formulaMake(dep + ' ~ ' + var + ' + ' + mediator)

        with localconverter(ro.default_converter + pandas2ri.converter):
            data = ro.conversion.py2rpy(data)

        if mediator in continuous:
            modelM = lm(MediationFormula, data) 
        else:
            modelM = glm(MediationFormula, data = data, family = "binomial")
        
        if dep in continuous:
            modelY = lm(OutcomeFormula, data) 
        else:
            modelY = glm(OutcomeFormula, data = data, family = "binomial")
        
        results = mediate(modelM, modelY, treat=var, mediator=mediator)
        dfR = summary(results)        
        capture(dfR, file = filePath)



def Mendelian_Ranomization(data, dep, indep, inst, path, control = list()):
    '''
    Conduct Mendelian Ranomization analysis using 2-Stage-Least-Square between 
    the dependent and independent variables passed based on the instrumental variables 
    sepcified. Write the results to Mendelian_Ranomization.txt in the specified path
    
    Args:
        data: DataFrame containing the items of interest
        dep: the dependent varaible in the analysis
        indep: the independent variable in the analysis
        control: the control variable(s) in the analysis -- can be a list
        inst: the instrumental variables to be used in the analysis
        path: Folder path to which the data will be saved

    Returns:
        an IV2SLS object containing the 2SLS analysis results
    '''
    if not os.path.exists(path + 'MendelianRandomization'):
        os.makedirs(path +'MendelianRandomization')
    currPath = path + 'MendelianRandomization/'
    
    MendAnalysis = open(currPath + 'Mendelian_Ranomization.txt', 'w')
    MendAnalysis.write('Mendelian Randomization Analysis Results: \n----------------------------------------------------------------------------------------\n\n')
    data = data.copy(deep = True)

    if type(control) == str:
        t = list()
        t.append(control)
        control = t

    if type(inst) == str:
        t = list()
        t.append(inst)
        inst = t

    formula = dep + ' ~'
    for var in control:
        formula += ' + C(' + str(var) +')'
    formula += ' + '
    for var in inst:
        formula += var + ' + '
    formula = formula[:-2]

    #Checking the first assumption -- is the instrument affecting the modifiable behavior?

    first_stage = smf.ols(formula, data=data).fit()
    for var in inst:
        MendAnalysis.write( var + " parameter estimate:, " + str(first_stage.params[var])+ '\n') 
        MendAnalysis.write(var + " p-value:, " + str(first_stage.pvalues[var]) + '\n')
        MendAnalysis.write('----------------------------------------------------------------------------------------\n')

    def parse(model, exog):
        param = model.params[exog]
        se = model.std_errors[exog]
        p_val = model.pvalues[exog]
        MendAnalysis.write(f"Parameter: {param}\n")
        MendAnalysis.write(f"SE: {se}\n")
        MendAnalysis.write(f"95 CI: {(-1.96*se,1.96*se) + param}\n")
        MendAnalysis.write(f"P-value: {p_val}\n")
        MendAnalysis.write('----------------------------------------------------------------------------------------\n')

    #Conducting Mendelian_Ranomization using 2 stage least square

    formula = dep + ' ~'
    for var in control:
        formula += ' + C(' + str(var) +')'
    formula += ' + [' + str(indep) + ' ~ '
    for var in inst:
        formula += var + '+'
    formula = formula[:-1] + ']'


    iv2sls = IV2SLS.from_formula(formula, data).fit()
    parse(iv2sls, exog = indep)
    MendAnalysis.close()

    return iv2sls


def Multivariate_Reg(data, path, target, continuous, categorical, correct = False, cutoff = 10, stepwise = True, VIF_values = list()):
    
    '''
    Conduct Multivariate Regression analysis between the target and independent variables
    passed. Write the results to Maltivariate_Reg.txt in statisticalAnalysis
    
    Args:
        data: DataFrame containing the items of interest
        path: Folder path to which the data will be saved
        target: the outcome varaible of interest in the analysis
        correct: boolean variable. If True, variables with high VIF value would be dropped
        cutoff: cutoff value to drop variables based on VIF.
        stepwise: if True, conduct stepwise ajdustment. Otherwise, the function won't.
        VIF_values: values for which checking for Multicolinearity using VIF is desired
            if left empty, the test would be applied to continuous variables. 
            Set to all to apply for all variables
        continuous: list containing continuous variables
        categorical: list of categorical variables

    Returns:
        The regression model before and after stepwise if stepwise is True
    '''
    data = data.copy(deep = True)

    indep = list(data)
    indep.remove(target)
    
    if not os.path.exists(path+'MultiVarRegression'):
        os.makedirs(path+'MultiVarRegression')
    currPath = path + 'MultiVarRegression/'

    MultiReg = open(currPath + 'MultivariateRegression.txt', 'w')


    #Check for linear relationship between features and target

    contPath = currPath+'ContinousVariablesCheck/'
    if not os.path.exists(contPath):
        os.makedirs(contPath)

    for var in continuous:
        xvalues, yvalues = list(), list()
        for j in range(0,100,1):
            i=j/100
            newdf=data.iloc[np.intersect1d(np.where(data[var]>i)[0],np.where(data[var]<i+0.01)[0]),:]
            xvalues.append(i+0.005)
            yvalues.append(np.mean(newdf[target]))

        plt.bar(xvalues,height=yvalues,width=0.005, linewidth = 1, edgecolor = '#000000')
        plt.xlabel(var)
        plt.ylabel(target)
        plt.savefig(contPath+var+'_barPlot.png')
        plt.close()

        sns.lmplot(x=var, y=target, data=data.sample(1000, random_state = 42), order=1)
        plt.ylabel(target)
        plt.xlabel(var)
        plt.savefig(contPath+var+'_scatterPlot.png')
        plt.close()

    catPath = currPath+'CategoricalVariablesCheck/'
    if not os.path.exists(catPath):
        os.makedirs(catPath)

    for var in categorical:
        xvalues=[]
        yvalues=[]
        stds=[]
        for i in range(data[var].nunique()):
            newdf=data.iloc[np.where(data[var]==i)[0],:]
            xvalues.append(i)
            yvalues.append(np.mean(newdf[target]))
            stds.append(2*np.std(newdf[target])/np.sqrt(len(newdf)))
        plt.bar(xvalues,height=yvalues,width=0.5)
        plt.errorbar(xvalues,yvalues,yerr=stds,fmt='o',markersize=1,capsize=8,color='r')
        plt.xlabel(var)
        plt.ylabel(target)
        plt.savefig(catPath+var+'_barPlot.png')
        plt.close()

        sns.lmplot(x=var, y=target, data=data.sample(1000, random_state = 42), order=1)
        plt.ylabel(target)
        plt.xlabel(var)
        plt.savefig(catPath+var+'_scatterPlot.png')
        plt.close()
        

    #Check for Multicolinearity
    
    vif_data = pd.DataFrame()
    if not VIF_values:
        VIF_values = continuous
    elif VIF_values == 'all' or VIF_values == ['all']:
        VIF_values = indep

    vif_data["feature"] = VIF_values
    vif_data["VIF"] = [variance_inflation_factor(data[VIF_values].values, i)
                        for i in range(len(VIF_values))]

    toDrop = list()        
    if correct:
        for i in range(len(vif_data)):
            if vif_data.iloc[i, 1] > cutoff:
                toDrop.append(vif_data.iloc[i,0])

        data.drop(toDrop, axis = 1, inplace = True)

    vifString = vif_data.to_string(header=True, index=False)
    
    MultiReg.write('VIF values without correction to check multicollinearity: \n\n')
    if not VIF_values:
        MultiReg.write('VIF check for Multicolinearity was not conducted'+'\n')
    else:
        MultiReg.write(vifString+'\n')
    MultiReg.write('----------------------------------------------------------------------------------------\n\n')

    if correct:
        for i in toDrop:
            vif_data = vif_data[vif_data.feature != i]
            indep.remove(i)
            data.drop([i], axis = 1, inplace = True)
        vifString = vif_data.to_string(header=True, index=False)
        MultiReg.write('VIF values after correction to check multicollinearity: \n\n')
        MultiReg.write(vifString+'\n')
        MultiReg.write('----------------------------------------------------------------------------------------\n')
    

    #Conduct regression analysis before stepwise adjustment
    X = sm.add_constant(data.drop([target], axis = 1)) 
    SMF_model = sm.OLS(endog= data[target], exog = X).fit()
    pvalues = SMF_model.pvalues
    corrected_p = fdr_correction(pvalues, alpha=0.05, method='indep')
    
    MultiReg.write('Multivariate Regression results before stepwise adjustment: \n\n')
    MultiReg.write(SMF_model.summary().as_text() + '\n\n')
    MultiReg.write('----------------------------------------------------------------------------------------\n')
    cp = [str(a) for a in corrected_p[1]]
    indep.insert(0, 'intercept')
    adjustedP = pd.DataFrame(cp, indep)
    MultiReg.write('adjusted p-values for Regression before stepwise adjustment: \n\n')
    MultiReg.write(adjustedP.to_string(header=False, index=True) + '\n\n')
    MultiReg.write('----------------------------------------------------------------------------------------\n')


    #Histogram & QQplot to test for normality before stepwise

    histo = sns.histplot(SMF_model.resid)
    fig = histo.get_figure()
    fig.savefig(currPath+"MultiRegHistogram-noStepwise.png")
    plt.close()
    fig, ax = plt.subplots(1, 1)
    sm.ProbPlot(SMF_model.resid).qqplot(line='s', color='#1f77b4', ax=ax)
    ax.title.set_text('QQ Plot')
    plt.savefig(currPath+"MultiRegQQPlot-noStepwise.png")
    plt.close()

    #Check for outliers before stepwise adjustment

    np.set_printoptions(suppress=True)
    influence = SMF_model.get_influence()
    cooks = influence.cooks_distance

    plt.scatter(np.arange(start=0, stop=len(cooks[0]), step=1), cooks[0])
    plt.xlabel('participant')
    plt.ylabel('Cooks Distance')
    plt.savefig(currPath+'MultiRegCooksDist-noStepwise.png')
    plt.close()
    

    #Check for Homoscedasticity using scale-location plot before stepwise adjustment
    y_predict = SMF_model.predict(X)

    fig, ax = plt.subplots(1, 1)
    
    sns.residplot(data[target],y_predict, lowess = True, scatter_kws={'alpha':0.5},
    line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
    ax.title.set_text('Scale Location')
    ax.set(xlabel='Fitted', ylabel='Standardized Residuals')
    plt.savefig(currPath+"MultiRegScaleLocation-noStepwise.png")
    plt.close()

    if stepwise:


        data.to_csv(currPath+'StepWiseRegData.csv')
        args = currPath + ' ' + str('linear') + ' ' + target
        os.system('Rscript ' + path + 'StepWiseRegression.R ' + args)
        os.remove(currPath + 'StepWiseRegData.csv')
        file = open(currPath+ 'stepWiseVars.txt')
        newVars = file.read().split()
        if '(Intercept)' in newVars:
            newVars.remove('(Intercept)')
      
        os.remove(currPath + 'stepWiseVars.txt')
        data = data[newVars + [target]]
    
        #Conduct regression analysis after stepwise adjustment

        X2 = sm.add_constant(data.drop([target], axis = 1)) 
        SMF_model2 = sm.OLS(endog= data[target], exog = X2).fit()
        pvalues2 = SMF_model2.pvalues
        corrected_p2 = fdr_correction(pvalues2, alpha=0.05, method='indep')

        MultiReg.write('Multivariate Regression resluts after stepwise adjustment: \n\n')
        MultiReg.write(SMF_model2.summary().as_text() + '\n\n')
        MultiReg.write('----------------------------------------------------------------------------------------\n')
        cp2 = [str(a) for a in corrected_p2[1]]
        newVars.insert(0, 'intercept')
        adjustedP2 = pd.DataFrame(cp2, newVars)
        MultiReg.write('adjusted p-values for Regression after stepwise adjustment: \n\n')
        MultiReg.write(adjustedP2.to_string(header=False, index=True) + '\n\n')
        MultiReg.write('----------------------------------------------------------------------------------------\n')


        #Histogram to test for normality after stepwise
        
        histo = sns.histplot(SMF_model2.resid)
        fig = histo.get_figure()
        fig.savefig(currPath+"MultiRegHistogram-Stepwise.png")
        plt.close()
        fig, ax = plt.subplots(1, 1)
        sm.ProbPlot(SMF_model2.resid).qqplot(line='s', color='#1f77b4', ax=ax)
        ax.title.set_text('QQ Plot')
        plt.savefig(currPath+"MultiRegQQPlot-Stepwise.png")
        plt.close()

        #Check for Homoscedasticity using scale-location plot after stepwise adjustment

        y_predict2 = SMF_model2.predict(X2)
        fig, ax = plt.subplots(1, 1)
        sns.residplot(data[target], y_predict2, lowess=True, scatter_kws={'alpha':0.5},
        line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
        ax.title.set_text('Scale Location')
        ax.set(xlabel='Fitted', ylabel='Standardized Residuals')
        plt.savefig(currPath+"MultiRegScaleLocation-Stepwise.png")
        plt.close()

        #Check for outliers after stepwise adjustment

        np.set_printoptions(suppress=True)
        influence = SMF_model2.get_influence()
        cooks = influence.cooks_distance

        plt.scatter(np.arange(start=0, stop=len(cooks[0]), step=1), cooks[0])
        plt.xlabel('participant')
        plt.ylabel('Cooks Distance')
        plt.savefig(currPath+'MultiRegCooksDist-Stepwise.png')
        plt.close()

        MultiReg.close()
        return SMF_model, SMF_model2
    
    MultiReg.close()
    return SMF_model

def UniVariateLogisitc(X, y, path, continuous, categorical):

    indep = X.name

    currPath = path
    univariate = open(currPath +'UnivariateLogisticRegression.txt', 'a')

    X = sm.add_constant(X)
    log_reg = GLM(y, X, family=families.Binomial()).fit()
    model_odds = pd.DataFrame(np.exp(log_reg.params), columns= ['Odds Ratio'])
    model_odds['p-value']= log_reg.pvalues
    model_odds[['2.5%', '97.5%']] = np.exp(log_reg.conf_int())

    univariate.write('Odds Ratios for univariate regression between ' + indep + ' and ' + y.name + ': \n\n')
    toWrite = model_odds.to_string(header = True, index = True)
    univariate.write(toWrite+'\n')
    univariate.write('----------------------------------------------------------------------------------------\n\n')
    univariate.close()

    log_reg.pvalues

    if indep in continuous or indep in categorical:
        predict = log_reg.predict(X)
        predict = np.log(predict)
        plt.scatter(X[indep], predict)
        plt.xlabel(indep)
        plt.ylabel('log odds of ' + str(y.name))
        plt.savefig(currPath+str(indep)+'-logOdds.png')
        plt.close()
    
    return log_reg.pvalues[1]


def Odds_Ratios(data, path, target, continuous, categorical = list(), correct = False, cutoff = 10, stepwise = True, VIF_values = list()):

    '''
    Calculates the odds ratio for the signifcant variables and write the results to Odds_Ratios.txt
    in statisticalAnalysis. 
    
    Args:
        data: DataFrame containing the items of interest
        path: Folder path to which the data will be saved
        target: the dependent varaible in the analysis
        correct: boolean variable. If True, variables with high VIF value would be dropped
        cutoff: Integer value for the cutoff value for VIF
        stepwise: if True, conduct stepwise ajdustment. Otherwise, the function won't.
        VIF_values: values for which checking for Multicolinearity using VIF is desired
            if left empty, the test would be applied to all continuous variables.
            set to all to apply for all the variables
        continuous: list containing continuous variables

    Returns:
        two np arrays containing the odds ratio as well as teh confidence intervals before
        and after doing stepwise adjustement if stepwise was True.
    '''
    data = data.copy(deep = True)

    indep = list(data)
    indep.remove(target)

    continuous = list()
    for col in data.drop([target],axis=1).columns:
        distinct = data[col].dropna().nunique()
        if distinct > 10:
            continuous.append(col)

    if not os.path.exists(path+'OddsRatios'):
        os.makedirs(path+'OddsRatios')
    currPath = path + 'OddsRatios/'

    if not os.path.exists(currPath + 'UnivariateLogistic/'):
            os.makedirs(currPath + 'UnivariateLogistic/')

    oddsRatio = open(currPath + 'Odds_Ratios.txt', 'w')

    #Checking for Multicolinearity

    vif_data = pd.DataFrame()
    if not VIF_values:
        VIF_values = continuous
    elif VIF_values == 'all' or VIF_values == ['all']:
        VIF_values = indep

    vif_data["feature"] = VIF_values
    vif_data["VIF"] = [variance_inflation_factor(data[VIF_values].values, i)
                        for i in range(len(VIF_values))]

    toDrop = list()        
    if correct:
        for i in range(len(vif_data)):
            if vif_data.iloc[i, 1] > cutoff:
                toDrop.append(vif_data.iloc[i,0])

        data.drop(toDrop, axis = 1, inplace = True)

    vifString = vif_data.to_string(header=True, index=False)
    
    oddsRatio.write('VIF values without correction to check multicollinearity: \n\n')
    oddsRatio.write(vifString+'\n')
    oddsRatio.write('----------------------------------------------------------------------------------------\n')

    if correct:
        for i in toDrop:
            vif_data = vif_data[vif_data.feature != i]
            indep.remove(i)
            data.drop([i], axis = 1, inplace = True)
        vifString = vif_data.to_string(header=True, index=False)
        oddsRatio.write('VIF values after correction to check multicollinearity: \n\n')
        oddsRatio.write(vifString+'\n')
        oddsRatio.write('----------------------------------------------------------------------------------------\n')
    
    
    #Find odds ratios before stepwise adjustment
    X = sm.add_constant(data.drop([target], axis = 1))
    log_reg = GLM(data[target], X, family=families.Binomial()).fit()
    model_odds = pd.DataFrame(np.exp(log_reg.params), columns= ['Odds Ratio'])
    model_odds['p-value']= log_reg.pvalues
    model_odds[['2.5%', '97.5%']] = np.exp(log_reg.conf_int())


    oddsRatio.write('Odds Ratios before stepwise adjustment: \n\n')
    toWrite = model_odds.to_string(header = True, index = True)
    oddsRatio.write(toWrite+'\n')
    oddsRatio.write('----------------------------------------------------------------------------------------\n')
    pvalues = log_reg.pvalues
    corrected_p = fdr_correction(pvalues, alpha=0.05, method='indep')
    cp = [str(a) for a in corrected_p[1]]
    indep.insert(0, 'intercept')
    adjustedP = pd.DataFrame(cp, indep)
    oddsRatio.write('adjusted p-values for Regression before stepwise adjustment: \n\n')
    oddsRatio.write(adjustedP.to_string(header=False, index=True) + '\n\n')
    oddsRatio.write('----------------------------------------------------------------------------------------\n')    
    indep.remove('intercept')
    #Checking for outliers before stepwise adjustement

    infl = log_reg.get_influence(observed=False)

    np.set_printoptions(suppress=True)
    cooks = infl.cooks_distance
    plt.scatter(np.arange(start=0, stop=len(cooks[0]), step=1), cooks[0])
    plt.xlabel('participant')
    plt.ylabel('Cooks Distance')
    plt.savefig(currPath+'LogitRegCooksDistance-noStepwise.png')
    plt.close()

    uniPath = currPath+ 'UnivariateLogisticRegression/'
    if not os.path.exists(uniPath):
        os.makedirs(uniPath)
    open(uniPath +'UnivariateLogisticRegression.txt', 'w').close()

    if stepwise: 
        data.to_csv(currPath+'StepWiseRegData.csv')
        args = currPath + ' ' + str('logit') + ' ' + target
        os.system('Rscript ' + path + 'StepWiseRegression.R ' + args)
        os.remove(currPath+'StepWiseRegData.csv')
        file = open(currPath+'stepWiseVars.txt')
        newVars = file.read().split()
        if '(Intercept)' in newVars:
            newVars.remove('(Intercept)')
        
        pVal = list()
        for var in newVars:
            pVal.append(UniVariateLogisitc(data[var], data[target], uniPath, continuous, categorical))
        
        corrected_p = fdr_correction(pVal, alpha=0.05, method='indep')
        cp = [str(a) for a in corrected_p[1]]

        adjustedPuni = pd.DataFrame(cp, indep)
        univariate = open(uniPath +'UnivariateLogisticRegression.txt', 'a')
        univariate.write('P-Values after adjustement for univariate regression: \n\n')
        toWrite = adjustedPuni.to_string(header = True, index = True)
        univariate.write(toWrite+'\n')
        
        os.remove(currPath+'stepWiseVars.txt')
        data = data[newVars + [target]]

        #Find odds ratios after stepwise adjustment

        X2 = sm.add_constant(data.drop([target], axis = 1))
        log_reg2 = GLM(data[target], X2, family=families.Binomial()).fit()

        model_odds2 = pd.DataFrame(np.exp(log_reg2.params), columns= ['Odds Ratio'])
        model_odds2['p-value']= log_reg2.pvalues
        model_odds2[['2.5%', '97.5%']] = np.exp(log_reg2.conf_int())

        oddsRatio.write('Odds Ratios after stepwise adjustment: \n\n')
        toWrite = model_odds2.to_string(header = True, index = True)
        oddsRatio.write(toWrite+'\n')
        oddsRatio.write('----------------------------------------------------------------------------------------\n')
        pvalues2 = log_reg2.pvalues
        corrected_p2 = fdr_correction(pvalues2, alpha=0.05, method='indep')
        cp2 = [str(a) for a in corrected_p2[1]]
        newVars.insert(0, 'intercept')
        adjustedP2 = pd.DataFrame(cp2, newVars)
        oddsRatio.write('adjusted p-values for Regression after stepwise adjustment: \n\n')
        oddsRatio.write(adjustedP2.to_string(header=False, index=True) + '\n\n')
        oddsRatio.write('----------------------------------------------------------------------------------------\n')    


        #Checking for outliers before stepwise adjustement

        infl2 = log_reg2.get_influence(observed=False)

        cooks2 = infl2.cooks_distance
        plt.scatter(np.arange(start=0, stop=len(cooks[0]), step=1), cooks2[0])
        plt.xlabel('participant')
        plt.ylabel('Cooks Distance')
        plt.savefig(currPath+'LogitRegCooksDistance-Stepwise.png')
        plt.close()

        oddsRatio.close()
        return model_odds, model_odds2
    
    else:
        pVal = list()
        for var in indep:
            pVal.append(UniVariateLogisitc(data[var], data[target], uniPath, continuous, categorical))
        
        corrected_p = fdr_correction(pVal, alpha=0.05, method='indep')
        cp = [str(a) for a in corrected_p[1]]

        adjustedPuni = pd.DataFrame(cp, indep)
        univariate = open(uniPath +'UnivariateLogisticRegression.txt', 'a')
        univariate.write('P-Values after adjustement for univariate regression: \n\n')
        toWrite = adjustedPuni.to_string(header = True, index = True)
        univariate.write(toWrite+'\n')


    oddsRatio.close()
    return model_odds


def oneWay_ANOVA(data, dep, indep, alpha, between, followUp, path):
    results = dict()
    if not os.path.exists(path+'oneWayANOVA'):
        os.makedirs(path+'oneWayANOVA')

    if not os.path.exists(path+'oneWayANOVA/'+indep):
        os.makedirs(path+'oneWayANOVA/'+indep)
    currPath = path+'oneWayANOVA/'+indep+'/'

    formula = dep + ' ~ C(' + indep + ')'
    oneWayANOVA = open(currPath+'oneWayANOVA_summary.txt', 'w')
    oneWayANOVA.write('----------------------------------------------------------------------------------------\n\n')
    
    #Create a box plot for outliers detection
    
    data = data[[indep, dep]]
    colors = ['#808080']
    box = sns.boxplot(x=indep, y=dep, data=data, palette=colors)
    fig = box.get_figure()
    fig.savefig(currPath+"oneWayANOVA_boxPlot.png")
    plt.close()
    

    #Create a bar plot

    sns.set(rc = {'figure.figsize':(15,10)})
    sns.set(font_scale = 1.5)
    sns.set_style('whitegrid')
    fig, bar = plt.subplots()
    
    colors = ['#808080']

    sns.barplot(x=indep, ax = bar, y=dep, data=data, palette = colors, capsize=.1)
    width = 0.3

    num_var2 = len(data[indep].unique())
    hatches = itertools.cycle(['+', 'x', '-', '/', '//', '\\', '*', 'o', 'O', '.'])

    '''
    for i, patch in enumerate(bar.patches):
        # Set a different hatch for each bar
        if i % num_var2 == 0:
            hatch = next(hatches)
        patch.set_hatch(hatch)
    '''

    for patch in bar.patches:
        current_width = patch.get_width()
        diff = current_width - width
        patch.set_width(width)
        patch.set_x(patch.get_x() + diff * .5)
        patch.set_edgecolor('#000000')

    fig = bar.get_figure()
    fig.savefig(currPath+"oneWayANOVA_barPlot.png")
    plt.close()

    #Conducting the ANOVA test
    oneWayANOVA.write('Results for one way ANOVA between ' + indep + ' and ' + dep + ' are: \n\n')
    res = stat()
    res.anova_stat(df=data, res_var=dep, anova_model=formula)
    asummary = res.anova_summary.to_string(header=True, index=True)
    oneWayANOVA.write(asummary + '\n')
    oneWayANOVA.write('----------------------------------------------------------------------------------------\n\n')
    results['ANOVA_Results'] = res.anova_summary

    #Follow-up Test TukeyTest
    if (res.anova_summary.iloc[0,4] > alpha) and (not followUp):
        oneWayANOVA.write('The p-value is lower than alpha; hence, no follow-up test was conducted\n')
    else:
        oneWayANOVA.write('Results for follow-up Tukey test between ' + indep + ' and ' + dep + ' are: \n\n')
        followUp = stat()
        followUp.tukey_hsd(df=data, res_var=dep, xfac_var=indep, anova_model=formula)
        fSummary = followUp.tukey_summary.to_string(header=True, index=True)
        oneWayANOVA.write(fSummary + '\n')
        results['Tukey_Results'] = followUp.tukey_summary
    oneWayANOVA.write('----------------------------------------------------------------------------------------\n\n')
        

    #histograms and QQ-plot for Normality detection
    sm.qqplot(res.anova_std_residuals, line='45')
    plt.xlabel("Theoretical Quantiles")
    plt.ylabel("Standardized Residuals")
    plt.savefig(currPath+'oneWayANOVA_qqPlot.png')
    plt.close()

    plt.hist(res.anova_model_out.resid, bins='auto', histtype='bar', ec='k') 
    plt.xlabel("Residuals")
    plt.ylabel('Frequency')
    plt.savefig(currPath+'oneWayANOVA_histogram.png')
    plt.close()

    #Shapiro-Wilk Test for Normality
    w, pvalue = stats.shapiro(res.anova_model_out.resid)
    oneWayANOVA.write('Results for Shapiro-Wilk test to check for normality are: \n\n')
    oneWayANOVA.write('w is: ' + str(w) + '/ p-value is: ' + str(pvalue) + '\n')
    oneWayANOVA.write('----------------------------------------------------------------------------------------\n\n')
    results['Shaprio-Wilk_Results'] = (w, pvalue)

    #Check for equality of varianve using Levene's test
    oneWayANOVA.write("Results for Levene's test to check for equality of variance are: \n\n")
    eqOfVar = stat()
    eqOfVar.levene(df=data, res_var=dep, xfac_var=indep)
    eqOfVarSummary = eqOfVar.levene_summary.to_string(header=True, index=False)
    oneWayANOVA.write(eqOfVarSummary + '\n')
    oneWayANOVA.write('----------------------------------------------------------------------------------------\n\n')
    results['Levene_Results'] = eqOfVar.levene_summary

    #The Kruskal-Wallis H test
    groups = list()
    vals = data[indep].unique()
    for val in vals:
        g = data.loc[data[indep] == val]
        g = g.loc[:, [dep]].squeeze().tolist()
        groups.append(g)

    Kruskal = stats.kruskal(*groups)
    oneWayANOVA.write('Results for the Kruskal-Wallis Test -- to be used if ANOVA assumptions are violated: \n\n')
    oneWayANOVA.write('statistic: ' + str(Kruskal[0]) + '/ p-value is: ' + str(Kruskal[1]) + '\n')
    oneWayANOVA.write('----------------------------------------------------------------------------------------\n\n')
    results['Kruskal-Wallis_Results'] = Kruskal

    
    #The dunn's test -- follow up
    if (Kruskal[1] > alpha) and (not followUp):
        oneWayANOVA.write('The p-value is lower than alpha; hence, no follow-up test was conducted for Kruskal test\n')    
    else:
        FSA = importr('FSA')
        dunnTest, formulaMaker = r['dunnTest'], r['as.formula']

        with localconverter(ro.default_converter + pandas2ri.converter):
            rDf = ro.conversion.py2rpy(data)

        formula = formulaMaker(dep + ' ~ ' + indep)
        dunnTwoWay = dunnTest(formula, data=rDf, method="bonferroni")

        asData, doCall, rbind = r['as.data.frame'], r['do.call'], r['rbind']
        dunnTwoWay = asData(doCall(rbind, dunnTwoWay))

        with localconverter(ro.default_converter + pandas2ri.converter):
            dunnTwoWay = ro.conversion.rpy2py(dunnTwoWay)

        dunnTwoWay.drop(['method', 'dtres'], inplace = True)

        for col in ['Z', 'P.unadj', 'P.adj']:
            dunnTwoWay[col] = pd.to_numeric(dunnTwoWay[col])
            dunnTwoWay[col] = np.round(dunnTwoWay[col], decimals = 5)
        
        oneWayANOVA.write("Results for follow-up Dunn's test between " + indep + " and " + dep + " are: \n\n")
        dunnSummary = dunnTwoWay.to_string(header=True, index=False)
        oneWayANOVA.write(dunnSummary + '\n\n')
      
    oneWayANOVA.write('----------------------------------------------------------------------------------------\n\n')

    
    return results


def twoWay_ANOVA(data, dep, indep, alpha, between, followUp, path):
        
    results = dict()
    if not os.path.exists(path+'twoWayANOVA'):
        os.makedirs(path+'twoWayANOVA')
    fname = indep[0] + '_' + indep[1]
    if not os.path.exists(path+'twoWayANOVA/'+fname):
        os.makedirs(path+'twoWayANOVA/'+fname)
    currPath = path+'twoWayANOVA/'+fname+'/'

    formula = dep + ' ~ C(' + indep[0] + ') + C(' + indep[1] + ') + C(' + indep[0] + '):C(' + indep[1] + ')'
    twoWayANOVA = open(currPath+'twoWayANOVA_summary.txt', 'w')
    twoWayANOVA.write('----------------------------------------------------------------------------------------\n\n')
        
    #Create a box plot
    data = data[indep + [dep]]
    colors = ['#808080', '#FFFFFF', '#C0C0C0']
    box = sns.boxplot(x=indep[0], y=dep, hue=indep[1], data=data, palette = colors, width = 0.6)
    fig = box.get_figure()
    fig.savefig(currPath+"twoWayANOVA_boxPlot.png")
    plt.close()

    #Create a bar plot
    sns.set(rc = {'figure.figsize':(15,10)})
    sns.set(font_scale = 1.5)
    sns.set_style('whitegrid')
    fig, bar = plt.subplots()
    
    colors = ['#808080', '#FFFFFF', '#C0C0C0']

    sns.barplot(x=indep[0], ax = bar, y=dep, hue=indep[1], data=data, 
    palette = colors, capsize=.1)
    width = 0.3

    num_var2 = len(data[indep[0]].unique())
    hatches = itertools.cycle(['+', 'x', '-', '/', '//', '\\', '*', 'o', 'O', '.'])

    '''
    for i, patch in enumerate(bar.patches):
        # Set a different hatch for each bar
        if i % num_var2 == 0:
            hatch = next(hatches)
        patch.set_hatch(hatch)
    '''

    for patch in bar.patches:
        current_width = patch.get_width()
        diff = current_width - width
        patch.set_width(width)
        patch.set_x(patch.get_x() + diff * .5)
        patch.set_edgecolor('#000000')

    bar.legend(frameon = 1, title = indep[1], fontsize = 15, title_fontsize = 20)     
    fig = bar.get_figure()
    fig.savefig(currPath+"twoWayANOVA_barPlot.png")
    plt.close()
        
    #Conducting the ANOVA test
    twoWayANOVA.write('Results for two way ANOVA between ' + indep[0] + '&' + indep[1] + ' and ' + dep + ' are: \n\n')
    res = stat()
    res.anova_stat(df=data, res_var=dep, anova_model=formula, ss_typ=3)
    asummary = res.anova_summary.iloc[1:, :].to_string(header=True, index=True)
    twoWayANOVA.write(asummary + '\n')
    twoWayANOVA.write('----------------------------------------------------------------------------------------\n\n')
    results['ANOVA_Results'] = res.anova_summary

    
    #Follow-up Test TukeyTest
    if (all(x > alpha  for x in res.anova_summary.iloc[1:4, 4].tolist())) and (not followUp):
        twoWayANOVA.write('All the p-values is lower than alpha; hence, no follow-up test was conducted\n\n')
    else:
        tukey = list()
        message = list()
        message.append('Main effect for ' + indep[0] + ':\n')
        message.append('Main effect for ' + indep[1] + ':\n')
        message.append('Interaction effect between ' + indep[0] + ' and ' + indep[1] + ':\n')                
        twoWayANOVA.write('Results for follow-up Tukey test between ' + indep[0] + ' & ' + indep[1] + ' and ' + dep + ' are: \n\n')
        followUp = stat()
        followUp.tukey_hsd(df=data, res_var=dep, xfac_var=indep[0], anova_model=formula)
        tukey.append(followUp.tukey_summary)
        followUp.tukey_hsd(df=data, res_var=dep, xfac_var=indep[1], anova_model=formula)
        tukey.append(followUp.tukey_summary)
        followUp.tukey_hsd(df=data, res_var=dep, xfac_var=indep, anova_model=formula)
        tukey.append(followUp.tukey_summary)
        for i in range(len(tukey)):
            fSummary = tukey[i].to_string(header=True, index=False)
            twoWayANOVA.write(message[i] + fSummary + '\n\n')
        results['Tukey_Results'] = tukey
    twoWayANOVA.write('----------------------------------------------------------------------------------------\n\n')
        
        
    #histograms and QQ-plot
    sm.qqplot(res.anova_std_residuals, line='45')
    plt.xlabel("Theoretical Quantiles")
    plt.ylabel("Standardized Residuals")
    plt.savefig(currPath+'twoWayANOVA_qqPlot.png')
    plt.close()

    plt.hist(res.anova_model_out.resid, bins='auto', histtype='bar', ec='k') 
    plt.xlabel("Residuals")
    plt.ylabel('Frequency')
    plt.savefig(currPath+'twoWayANOVA_histogram.png')
    plt.close()
        
        
    #Shapiro-Wilk Test for Normality
    w, pvalue = stats.shapiro(res.anova_model_out.resid)
    twoWayANOVA.write('Results for Shapiro-Wilk test to check for normality are: \n\n')
    twoWayANOVA.write('w is: ' + str(w) + '/ p-value is: ' + str(pvalue) + '\n')
    twoWayANOVA.write('----------------------------------------------------------------------------------------\n\n')
    results['Shaprio-Wilk_Results'] = (w, pvalue)
        
    #Check for equality of varianve using Levene's test
    twoWayANOVA.write("Results for Levene's test to check for equality of variance are: \n\n")
    eqOfVar = stat()
    eqOfVar.levene(df=data, res_var=dep, xfac_var=indep)
    eqOfVarSummary = eqOfVar.levene_summary.to_string(header=True, index=False)
    twoWayANOVA.write(eqOfVarSummary + '\n')
    twoWayANOVA.write('----------------------------------------------------------------------------------------\n\n')
    results['Levene_Results'] = eqOfVar.levene_summary

    
    #The scheirer Ray Hare test

    rcompanion = importr('rcompanion')
    formulaMaker = r['as.formula']
    scheirerRayHare = r['scheirerRayHare']

    formula = formulaMaker(dep + ' ~ '  + indep[0] + ' + ' + indep[1])
    
    with localconverter(ro.default_converter + pandas2ri.converter):
            rDf = ro.conversion.py2rpy(data)

    scheirerANOVA = scheirerRayHare(formula, data = rDf)

    with localconverter(ro.default_converter + pandas2ri.converter):
        scheirerANOVA = ro.conversion.rpy2py(scheirerANOVA)


    twoWayANOVA.write('Results for the scheirer Ray Hare Test -- to be used if ANOVA assumptions are violated: \n\n')
    scheirerSummary = scheirerANOVA.to_string(header=True, index=True)
    twoWayANOVA.write(scheirerSummary + '\n')
    twoWayANOVA.write('----------------------------------------------------------------------------------------\n\n')
    results['scheirerRayHare'] = scheirerANOVA
    
    #The dunn's test -- follow up
    if (all(x > alpha  for x in scheirerANOVA['p.value'].tolist())) and (not followUp):
        twoWayANOVA.write('All the p-values is lower than alpha; hence, no follow-up test was conducted for ScheirerRayHare test\n\n')
    
    else:
        FSA = importr('FSA')
        dunnTest = r['dunnTest']
        data['interaction'] = data[indep[0]].astype(str) + '_' + data[indep[1]].astype(str)
        with localconverter(ro.default_converter + pandas2ri.converter):
            rDf = ro.conversion.py2rpy(data)

        formula = formulaMaker(dep + ' ~ interaction')
        dunnTwoWay = dunnTest(formula, data=rDf, method="bonferroni")

        asData, doCall, rbind = r['as.data.frame'], r['do.call'], r['rbind']
        dunnTwoWay = asData(doCall(rbind, dunnTwoWay))

        with localconverter(ro.default_converter + pandas2ri.converter):
            dunnTwoWay = ro.conversion.rpy2py(dunnTwoWay)

        dunnTwoWay.drop(['method', 'dtres'], inplace = True)
        for col in ['Z', 'P.unadj', 'P.adj']:
            dunnTwoWay[col] = pd.to_numeric(dunnTwoWay[col])
            dunnTwoWay[col] = np.round(dunnTwoWay[col], decimals = 5)
        
        twoWayANOVA.write("Results for follow-up Dunn's test between " + indep[0] + ' & ' + indep[1] + " and " + dep + " are: \n\n")
        dunnSummary = dunnTwoWay.to_string(header=True, index=False)
        twoWayANOVA.write(dunnSummary + '\n\n')
      
    twoWayANOVA.write('----------------------------------------------------------------------------------------\n\n')
    
    return results


def ANOVA(data, path, dep, indep, alpha = 0.05, oneWay = True, followUp = False):
    '''
    Conduct an ANOVA analysis -- either one or two way -- between the dependent and independent variables
    passed. If there is signifcant effect found, conduct a follow up test. The function checks for the ANOVA
    assumption and provide alternative tests such as Kruskal-Wallis H. Results will be stored at ###
    
    Args:
        data: DataFrame containing the items of interest
        path: Folder path to which the data will be saved
        dep: the dependent varaible in the analysis
        indep: column names in the data frame containing the groups -- can be 
            a string or a list of two strings
        alpha: minimum value for the p-value for the effect to be signifcant
            conduct repeated measures ANOVA -- to be implemented.
        oneWay: if True, conduct one way ANOVA. if Fasle, conduct two way ANOVA.
        followUp: if True, a follow up test would be conducted regardless of the ANOVA p-value

    Returns:
        a dictionary mapping each test conducted to its results
    '''
    data = data.copy(deep = True)
    if oneWay:
        if (type(indep) == list):
            indep = indep[0]
        return oneWay_ANOVA(data, dep, indep, alpha, False, followUp, path)
    else:
        return twoWay_ANOVA(data, dep, indep, alpha, False, followUp, path)


def Association_Analysis(data, path, vars, oneTarget = False, target = '', chi = False, fisher = False):
        
    ''' 
    Do Chi Square test, Fisher exact test, and g-test of independence between the passed variables.
    Write the findings to to Association_Analysis.txt in statisticalAnalysis.
        
    Args:
        data: DataFrame containing the items of interest
        path: Folder path to which the data will be saved
        var: list of variables to apply the tests to. Have to contain at least two
        oneTarget: if True, the analysis would be done against the specified variable.
            If True, the parameter target must be passed
        target: the variable of interest to test other vraiales against
        chi: if True, conduct ChiSquare test
        Fisher: if True, conduct Fisher exact test -- can result in an error

    Returns:
        a data frame containing the results for chi-square, fisher, and g tests
    '''
    data = data.copy(deep = True)
    if not os.path.exists(path+'AssociationAnalysis'):
        os.makedirs(path+'AssociationAnalysis')
    currPath = path + 'AssociationAnalysis/'
    
    #results = pd.DataFrame(columns=['var1-var2','Pearson-Chi-square', 'Chi-p-value', "Chi-Cramer's-phi",
    #    'Fisher-Odds-ratio', 'Fisher-2-sided-p-value', "Fisher-Cramer's-phi", 'G-Test-Log-likelihood-ratio', 
    #    'G-Test-p-value', "G-Test-Cramer's-phi"])
    
    resultsG = pd.DataFrame(columns=['var1-var2', 'G-value', 'G-df', 'G-pValue', 'expectedAbove5'])
    resultsFisher = pd.DataFrame(columns=['var1-var2', 'Fisher-pValue', 'Fisher-oddsRatio'])
    resultsChi = pd.DataFrame(columns=['var1-var2', 'Chi-value', 'Chi-df', 'Chi-pValue'])
    belowThreshhold = list()

    rows = list()
    tempList = list(vars)
    for var1 in vars:
        print(var1)
        if oneTarget and var1!=target:
            #fisher = FisherExactAnalysis(data, var1, target)
            g, exp = G_TestAnalysis(data, var1, target)

            rows.append([str(var1)+'-'+str(target), 
            g[0][0], g[1][0], g[2][0], exp])
            if exp < 0.8:
                belowThreshhold.append(var1, target)

        elif not oneTarget:
            tempList.remove(var1)
            for var2 in tempList:
                print(var1, var2)
                g, exp = G_TestAnalysis(data, var1, var2)
                rows.append([str(var1)+'-'+str(var2), 
                g[0][0], g[1][0], g[2][0], exp])
                if exp < 0.8:
                    belowThreshhold.append((var1, var2))
        
        
        
    for i in range(len(rows)):
        resultsG.loc[len(resultsG.index)] = rows[i]
    
    AssociationAnalysis = open(currPath+'AssociationAnalysisGTest.txt', 'w')
    AssociationAnalysis.write('Association Analysis Results of G Test: \n---------------------------------------------------\n\n')
    for i in range(len(resultsG)):
        two_var = resultsG.iloc[i, :]
        two_var = two_var.to_frame()
        variables = str(two_var.iloc[0,0]).split('-')
        two_var = two_var.iloc[1: , :]
        AssociationAnalysis.write('The Association Analysis Results for G Test between ' +variables[0]+' and ' + variables[1] + ' are: \n\n')
        toWrite = two_var.to_string(header = False, index = True)
        AssociationAnalysis.write(toWrite+'\n')
        AssociationAnalysis.write('----------------------------------------------------------------------------------------\n\n')
        
    AssociationAnalysis.write('Variables that do not fullfil the GTest and Chi Square assumptions: \n\n')
    for var in belowThreshhold:
        AssociationAnalysis.write(str(var)+'\n')
    AssociationAnalysis.write('----------------------------------------------------------------------------------------\n\n')

    
    AssociationAnalysis.close()
    results = resultsG.copy(deep = True)

    if fisher:
        rows = list()
        tempList = list(vars)
        for var1 in vars:
            if oneTarget and var1!=target:
                fisher, _ = FisherExactAnalysis(data, var1, target)
                rows.append([str(var1)+'-'+str(target), 
                fisher[0][0], fisher[2][0]])

            elif not oneTarget:
                tempList.remove(var1)
                for var2 in tempList:
                    fisher, _ = FisherExactAnalysis(data, var1, var2)
                    rows.append([str(var1)+'-'+str(var2), 
                    fisher[0][0], fisher[2][0]])
            
        for i in range(len(rows)):
            resultsFisher.loc[len(resultsFisher.index)] = rows[i]
        
        AssociationAnalysis = open(currPath+'AssociationAnalysisFisherTest.txt', 'w')
        AssociationAnalysis.write('Association Analysis Results of Fisher Test: \n---------------------------------------------------\n\n')
        for i in range(len(resultsFisher)):
            two_var = resultsFisher.iloc[i, :]
            two_var = two_var.to_frame()
            variables = str(two_var.iloc[0,0]).split('-')
            two_var = two_var.iloc[1: , :]
            AssociationAnalysis.write('The Association Analysis Results for Fisher Test between ' +variables[0]+' and ' + variables[1] + ' are: \n\n')
            toWrite = two_var.to_string(header = False, index = True)
            AssociationAnalysis.write(toWrite+'\n')
            AssociationAnalysis.write('----------------------------------------------------------------------------------------\n\n')
            
        
        AssociationAnalysis.close()
        results = pd.concat([results, resultsFisher], axis = 1)

    if chi:
        rows = list()
        tempList = list(vars)
        for var1 in vars:
            if oneTarget and var1!=target:
                chiTest, _ = Chi_SquareAnalysis(data, var1, target)
                rows.append([str(var1)+'-'+str(target), 
                chiTest[0][0], chiTest[1][0], chiTest[2][0]])

            elif not oneTarget:
                tempList.remove(var1)
                for var2 in tempList:
                    chiTest, _ = Chi_SquareAnalysis(data, var1, var2)
                    rows.append([str(var1)+'-'+str(var2), 
                    chiTest[0][0], chiTest[1][0], chiTest[2][0]])
            
        for i in range(len(rows)):
            resultsChi.loc[len(resultsChi.index)] = rows[i]
        
        AssociationAnalysis = open(currPath+'AssociationAnalysisChiSquareTest.txt', 'w')
        AssociationAnalysis.write('Association Analysis Results of Chi Square Test: \n---------------------------------------------------\n\n')
        for i in range(len(resultsG)):
            two_var = resultsChi.iloc[i, :]
            two_var = two_var.to_frame()
            variables = str(two_var.iloc[0,0]).split('-')
            two_var = two_var.iloc[1: , :]
            AssociationAnalysis.write('The Association Analysis Results for Chi Square Test between ' +variables[0]+' and ' + variables[1] + ' are: \n\n')
            toWrite = two_var.to_string(header = False, index = True)
            AssociationAnalysis.write(toWrite+'\n')
            AssociationAnalysis.write('----------------------------------------------------------------------------------------\n\n')
            
        
        AssociationAnalysis.close()
        results = pd.concat([results, resultsChi], axis = 1)


    return results


def Chi_SquareAnalysis(data, var1, var2):
    table = np.array(pd.crosstab(data[var1], data[var2], margins = False))
    chi = r['chisq.test']
    res = chi(table)
    return res, None


def FisherExactAnalysis(data, var1, var2):
    table = np.array(pd.crosstab(data[var1], data[var2], margins = False))
    stats = importr('stats')
    res = stats.fisher_test(table)
    return res, None

    #_, test_results_Fisher, _ = rp.crosstab(data[var1], data[var2],test= "fisher", 
    #expected_freqs= True, prop= "cell", correction = True)
    #return test_results_Fisher

def G_TestAnalysis(data, var1, var2):

    _, _, exp = rp.crosstab(data[var1], data[var2],test= "g-test", 
    expected_freqs= True, prop= "cell", correction = True)

    exp = np.array(exp)

    total = exp.shape[0]*exp.shape[1]
    overFive = 0
    for i in range(len(exp)):
        for j in range(len(exp[0])):
            if exp[i][j] >= 5:
                overFive +=1
        
    table = np.array(pd.crosstab(data[var1], data[var2], margins = False))
    desk = importr('DescTools')
    if (table.shape == (2,2)):
        res = desk.GTest(table, correct = 'yates')
    else:
        res = desk.GTest(table, correct = 'williams')

    return res, overFive/total
