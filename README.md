# UKKB_Analyzer
A package used to run Machine Learning as well as Statical Analysis on UK_BioBank data

## Requirements
  - Lorem ipsum dolor sit amet, consectetur adipiscing elit. Suspendisse felis est, porttitor nec feugiat eget, porta vel nulla. 
    Duis venenatis, magna in semper consectetur, magna odio aliquet leo, et aliquam diam enim at diam.
  - hasellus facilisis orci elit, sed eleifend orci porttitor quis. In quis nisi non nulla laoreet elementum. Aliquam purus tellus,
    lobortis in auctor cursus sit amet sapien.

## Files Breakdown:
### UKBB_Analyzer.py
  - The main file of the package that will be used to run all the different analyses.
  - This should be the only file that NEED to be modified to use the package -- you're welcome to navigate the other files,
    but the main functionalities should be easily accessible from here.
  - For now, the available analysis to be conducted are Feature Selection, Classification, and Statistical Analysis. 
    Feature Selection & Classification can be run by calling NormalRun in kFoldValidationRuns class. To conduct Statistical Analysis, 
    create a Stats object and call the appropriate function with the appropriate data, detailed descriptions of that 
    can be found below description of StatisitcalAnalysis.py
    
#### Function: main()
  - calls other functions to do analyze the data
  
### kFoldValidationRuns.py
  - main getway to run the feature selection and classifications
  - depending on the parameters passed will run feature selection, classification, or both
  - The feature selection and classification runs are parelized as they take a long time. If the parallelization 
    is causing issues, it probably can be traced back to this file.
  - Note: the data that gets passed to this file to be used for classification and feature selection has to be
    one-hot-encoded with categorical outcome
 
#### Function: NormalRun()
  - runs either feature selection, classification, or both
  - attributes are broken down in the function header
  
### RunFeatureSelection.py
  - the file running the feature selection with bootstrapping
  - the resutls of these runs are saved in xxxxx
  - Note: this file doesn't combine the results of the bootstrap runs -- this happens in ClassFeatureStats.py
  
#### Function: fselectNew()
  - main function to run the feature selection with bootstrapping
  - the function creates a list of tuples that contains the input for the helper function used to run the feature selections
  - the tuples mainly contain the feature selection method and a number ranging from 1 to fselectRepeat passed to NormalRun, 
    which would represent the random_state used to resample the data for bootstrapping
  - attributes are broken down in the function header
  
#### Function: fsNewHelper()
  - helper function for fselectNew() that takes a tuple and does data resampling based on the content of the tuple.
  - After resampling, important features are found for that run based on the feature selection method passed 
    and the output is saved in a txt file in xxxxx
  - The function that does the selection of the indices of important features is run_feature_selection, which is called by fsNewHelper
  - attributes are broken down in the function header

#### Function: run_feature_selection()
  - this function receives X and y which would be predictive features and outcome variable as well as a string 
    representing the feature selection method
  - based on the string representing the feature selection method, another function would be called with X and y, 
    which will return the indices of the selected features
    
 #### Function: fisher_exact_test(), cfs(), merit_calculation(), fcbf(), reliefF(), infogain(), fcbfHelper(), jmi(), mrmr(), chisquare(), su_calculation(), mutual_information(), conditional_entropy(), entropy()
  - These are the functions doing the feature selection or helper functions for them
  - There should be no need to modify any of these functions, but they include detailed description of what they are doing 
    and the parameters needed in the python code
    
### RunClassifiers.py
  - the file running each of the specified classifiers with each of the feature selection methods to produce a heatmap
  - the resutls, which include the confusion matrices, of these runs are saved in xxxxx
  - Note: this file doesn't produce the heatmaps -- this happens in ClassFeatureStats.py

#### Function: classify()
  - The main function of the file -- gets called to run the classification'
  - The function recieves a combination of classification and feature selection method as well as the data.
  - The function breaks down the data into train and testing, and then KFolds cross validation is being conducted on the train
    data, breaking it to train and validation sets. on each of the KFold runs, Baysian search is used for hyperparameter optimization
  - After finding the best parameters (depending on optimizing F1 or Accuracy which can be changed), the classifier is being run
    on the test data that was split before running KFold cross validation.
  - This whole processing of train/validation/test split is being repeated n_seed times (n_seed gets passed in UKBB_Analyzer)
  - attributes are broken down in the function header

#### Function: getParameters()
  - A helper function for classify() that return a dictionary with hyperparameters and the values that Baysian search
    should look through for the optimization
    
#### Sub-Class: MyClassifier()
  - A sub class that will be used to run the classification runs that gets called from the Baysian search function
  - The parameter includes an estimator, based on which an instance of the appropratie classifier will be created, and 
    a method, based on which a subset of features chosed by that method (fselect method) will be used.
  - The class has default values for all the hyperparameters used by all the classifiers, of which the appropraite ones are
    used to create instance of a new classifier.
  - After that, set_params() function is used to update these parameters during each run of the Baysian search
  - Note: what is being optimized in these runs is what's being returned by the score() function, and if a differnt scoring
    metric is desired, it should be calculated using standrded python libraries and returned (almost every scoring metric can
    be calculated by only using y and yp -- which are the predicted values for y)
  - All the functions in the class are either writing already exisiting functions of the classifiers to do the same 
    job but with some modification to fit the code -- detailed description of each of the functions is included in the code


















## Association Rule Learning: (undone)
  - Input must be only one hot encoded and outcome variables has to be categorized
  - can't include continuous variables

## Mediation Analysis: (undone)
  - (Not 100% sure) Outcome preferably continuous
  - mediator can be either continuous or categorical (doesn't need to be one hot encoded)
  - independent can be either continuous or categorical (doesn't need to be one hot encoded)

## Mendelian Randomization (undone)
  - Don't remember honestly but I think continuous and categorical non-one-hot-encoded variables are fine

## Multivariate Regression (undone)
  - categorical variables need to be one hot encoded
  - continuous variables are okay
  - outcome variable has to be continuous

## Logistic Regression (Odds Ratios) (undone)
  - categorical variables need to be one hot encoded
  - continuous variables are okay
  - outcome variable has to be binary or categorical

## ANOVA (undone)
  - indep variables have to be categorical (preferably not one hot encoded)
  - outcome variable has to be continuous

## Association Analysis (G-Test) (undone)
  - indep variables have to be categorical (preferably not one hot encoded)
  - outcome variable has to be continuous
  - Don't include interaction terms for this analysis
