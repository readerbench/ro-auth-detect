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
dataFT <- read.xlsx("data/10A-FT-predictions.xlsx", 1, header = TRUE)
head(dataFT)
str(dataFT)

dataFT$Model=factor(dataFT$Model, levels=c("EL", "BERT", "Hybrid"))
dataFT$Correct=factor(dataFT$Correct, levels=c("Yes", "No"))               

xtabs(~ Model + Correct,data=dataFT)

tableFT = xtabs( ~ Correct + Model,
               data=dataFT)

barplot(tableFT,
        beside = TRUE,
        legend = TRUE,
        ylim = c(0,50),   
        cex.names = 0.8,   
        cex.axis = 0.8,    
        args.legend = list(x   = "topright",
                           cex = 0.8,
                           bty = "n"))

cochran.qtest(Correct ~ Model | Index,
              data = dataFT)

pairwiseMcnemar(Correct ~ Model | Index,
                     data = dataFT,
                     digits = 3)

