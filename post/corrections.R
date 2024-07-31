#
# C O R R E C T I O N  A N A L Y S I S
#

library(dplyr)
library(ggplot2)


minorities <- c("0.20", "0.38")
techniques <- c("smote", "adasyn")

for (minority in minorities) {
  for (technique in techniques){ 
    filenameBefore = paste("before-", technique, "-", minority, "-correction", ".csv", sep="")
    filenameAfter = paste("after-", technique, "-", minority, "-correction", ".csv", sep="")
    before <- read.csv(filenameBefore)
    after <- read.csv(filenameAfter)
    
    p <- ggplot(before, aes(x=before$hue, y=before$type)) + 
      geom_boxplot()
    p
  }
}
