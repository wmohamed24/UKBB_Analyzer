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


## Association Rule Learning: (done)
  - Input must be only one hot encoded and outcome variables has to be categorized
  - can't include continuous variables

## Mediation Analysis: (undone)
  - (Not 100% sure) Outcome preferably continuous
  - mediator can be either continuous or categorical (doesn't need to be one hot encoded)
  - independent can be either continuous or categorical (doesn't need to be one hot encoded)

## Mendelian Randomization (undone)
  - Don't remember honestly but I think continuous and categorical non-one-hot-encoded variables are fine

## Multivariate Regression
  - categorical variables need to be one hot encoded
  - continuous variables are okay
  - outcome variable has to be continuous

## Logistic Regression (Odds Ratios)
  - categorical variables need to be one hot encoded
  - continuous variables are okay
  - outcome variable has to be binary or categorical

## ANOVA
  - indep variables have to be categorical (preferably not one hot encoded)
  - outcome variable has to be continuous

## Association Analysis (G-Test)
  - indep variables have to be categorical (preferably not one hot encoded)
  - outcome variable has to be continuous
  - Don't include interaction terms for this analysis
