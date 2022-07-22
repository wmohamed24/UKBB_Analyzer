# UKKB_Analyzer
A package used to run Machine Learning as well as Statical Analysis on UK_BioBank data


# Association Rule Learning:
  - Input must be only one hot encoded and outcome variables has to be categorized
  - can't include continuous variables

# Mediation Analysis:
  - (Not 100% sure) Outcome preferably continuous
  - mediator can be either continuous or categorical (doesn't need to be one hot encoded)
  - independent can be either continuous or categorical (doesn't need to be one hot encoded)

# Mendelian Randomization
  - Don't remember honestly but I think continuous and categorical non-one-hot-encoded variables are fine

# Multivariate Regression
  - categorical variables need to be one hot encoded
  - continuous variables are okay
  - outcome variable has to be continuous

# Logistic Regression (Odds Ratios)
  - categorical variables need to be one hot encoded
  - continuous variables are okay
  - outcome variable has to be binary or categorical

# ANOVA
  - indep variables have to be categorical (preferably not one hot encoded)
  - outcome variable has to be continuous

# Association Analysis (G-Test)
  - indep variables have to be categorical (preferably not one hot encoded)
  - outcome variable has to be continuous
  - Don't include interaction terms for this analysis
