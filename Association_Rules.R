
#install.packages("arules", repos='http://cran.us.r-project.org')
#install.packages("arulesViz", repos='http://cran.us.r-project.org')
#install.packages("RColorBrewer", repos='http://cran.us.r-project.org')
#install.packages("DescTools")
#install.packages("FSA")


library(arules)
library(arulesViz)
library(RColorBrewer)
library(htmlwidgets)


args <- commandArgs(trailingOnly = TRUE)

path = '/Users/wael/Desktop/AyLab/MoodDisorderResearch/StatisticalAnalysis/Apriori/'
#path <- args[1]
fname <- paste(path,'AprioriData.csv',sep="")
data <- read.csv(fname)
data <- data[,!names(data) %in% c("X")]
data <- data.frame(lapply(data, as.logical))


sup <- 0.005 #as.numeric(args[2])
con <- 0.05 #as.numeric(args[3])
max <- 5 #as.numeric(args[4])
min <- 2 #as.numeric(args[5])
varOfInterest <- 'GAD7_1' #args[6]


if (varOfInterest != 'none'){
  rules <- apriori(data, parameter = list(supp=sup, conf=con, maxlen=max, minlen = min, target ="rules"),
                   appearance = list(default="lhs",rhs=varOfInterest))
} else {
  rules <- apriori(data, parameter = list(supp=sup, conf=con, maxlen=max, minlen = min, lift = 3, target ="rules"))
}

rules <- sort(rules, decreasing = TRUE, na.last = NA, by = "lift")
rules <- subset(rules, subset = lift > 1.9)


pdf(file = paste(path, 'AprioriMatrixBased.pdf', sep = ""))
plot(rules, method = "grouped")
dev.off()

rules.df = DATAFRAME(rules)
write.csv(rules.df,paste(path,"apriori.csv",sep=""), row.names = FALSE)


saveWidget(plot(rules, method = "graph",  engine = "htmlwidget"), 
           file= paste(path,"AprioriNetwork.html",sep=""))
