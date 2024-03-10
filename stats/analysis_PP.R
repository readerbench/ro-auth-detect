my_packages <- c("psych", "xlsx", "RVAideMemoire", "coin", "reshape2", "rcompanion") # Specify packages
not_installed <- my_packages[!(my_packages %in% installed.packages()[ , "Package"])]  # Extract not installed packages
if(length(not_installed)) install.packages(not_installed) # Install not installed packages
lapply(my_packages, require, character.only = TRUE) # Load multiple packages

# Initial cleaning and environment configuration
rm(list=ls(all=TRUE))
getwd()
list.files()
options(digits=4)

################################################################################
# Data loading
dataPP <- read.xlsx("data/10A-PP-predictions.xlsx", 1, header = TRUE)
head(dataPP)
str(dataPP)

dataPP$Model=factor(dataPP$Model, levels=c("EL", "BERT", "Hybrid"))
dataPP$Correct=factor(dataPP$Correct, levels=c("Yes", "No"))               

xtabs(~ Model + Correct,data=dataPP)

tablePP = xtabs( ~ Correct + Model,
               data=dataPP)

barplot(tablePP,
        beside = TRUE,
        legend = TRUE,
        ylim = c(0, 700),   
        cex.names = 0.8,   
        cex.axis = 0.8,    
        args.legend = list(x   = "topright",
                           cex = 0.8,
                           bty = "n"))

cochran.qtest(Correct ~ Model | Index,
              data = dataPP)

pairwiseMcnemar(Correct ~ Model | Index,
                     data   = dataPP,
                     digits = 3)

