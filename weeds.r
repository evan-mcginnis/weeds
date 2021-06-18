#
# W E E D S
#


library(caTools)

setwd("c:/University of Arizona/weeds")

# Read in the LR Data
myData = read.csv("lr.csv", header=TRUE)
# Drop bits we don't care about
df = subset(myData, select= -c(name, number))
df["type"][df["type"] == "1"] <- "0"
df["type"][df["type"] == "2"] <- "1"
