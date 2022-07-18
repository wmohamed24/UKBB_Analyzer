#install.packages('tidyverse', dependencies = TRUE, repos='http://cran.us.r-project.org')
#install.packages('caret', dependencies = TRUE, repos='http://cran.us.r-project.org')
#install.packages('leaps', dependencies = TRUE, repos='http://cran.us.r-project.org')
#install.packages("My.stepwise", dependencies = TRUE, repos='http://cran.us.r-project.org')

library(MASS)
library(tidyverse)
library(caret)
library(leaps)
library(readr)

#library(My.stepwise)

args = commandArgs(trailingOnly = TRUE)

path = args[1]
fname = paste(path,'StepWiseRegData.csv',sep="")
regType = args[2]
var = args[3]


data = read.csv(fname)
data = data[,!names(data) %in% c("X")]

if (regType == 'linear'){
  full.model <- lm(GAD7 ~., data = data)
  step.model <- stepAIC(full.model, direction = "both", 
                        trace = FALSE)
  #b <- regsubsets(GAD7 ~ ., data=data, nbest=1, nvmax=30, really.big=T)
  #print(summary(b))

  
} else if (regType == 'logit') {
  full.model <- glm(GAD7_1 ~ ., data = data, family = "binomial")
  #step.model <- stepAIC(full.model, direction = "both", 
  #                     trace = FALSE)
  step.model <- stepAIC(full.model, direction = "both", 
                        trace = FALSE)
  #step.model <- stepwise(full.model, direction="forward/backward", criterion = "AIC")
}


newModel <- model.matrix(step.model)
newVars <- as.list(colnames(newModel)) 
lapply(newVars, write, paste(path,'stepWiseVars.txt',sep=""), append=TRUE, ncolumns=1000)


