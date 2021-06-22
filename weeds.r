#
# W E E D S
#


library(caTools)

setwd("c:/University of Arizona/weeds")

# Read in the LR Data
myData = read.csv("lr.csv", header = TRUE)


# 
# C L E A N  U P  D A T A 
#
# Drop bits we don't care about
#df = subset(myData, select= -c(name, number))
# Replace values so we have some that correspond to true/false
#df["type"][df["type"] == "1"] <- "0"
#df["type"][df["type"] == "2"] <- "1"

#final = as.vector(df)
#
# S P L I T
#
# Unless you give the split one column name, it will split 50/50
split = sample.split(myData$type, SplitRatio = 0.6)
train <- subset(myData, split == "TRUE")
test <- subset(myData, split == "FALSE")
myData$type <- as.factor(myData$type)

weedModel <- glm(type ~ ratio + shape + size, data = train, family='binomial', maxit=100 )
#weedModel <- glm(type ~ ., data = train, family='binomial' )
summary(weedModel)

res <- predict(weedModel, test, type="response")
res

res <- predict(weedModel, train, type="response")
res

# 
# Confusion Matrix
#
confusion <- table(Actual_value=train$type, Predicted_value= res > 0.5)
confusion

