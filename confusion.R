#
# C O N F U S I O N  M A T R I X
#

library(optparse)
library(tidymodels)
library(ggplot2)
library(stringr)
library(gsubfn)
library(cowplot)

# RStudio is a bit of a pain here.  I can't specify run-time parameters, so I suppose I should
# just make these optional with some defaults so I can debug.

options <- list(
  make_option(c("-i", "--input", type="character", default="results.csv", help="Input CSV", metavar="character")),
  make_option(c("-o", "--output", type="character", default="figures", help="Output directory for plots", metavar="character"))
)

optParser <- OptionParser(option_list=options)
opt = parse_args(optParser)

RESULT_RESULT <- "results"
RESULT_TECHNIQUE <- "technique"

input <- "results.csv"
output <- "figures"

#resultsFiles = c("results-forest.csv", "results-logistic.csv", "results-decision.csv", "results-gradient.csv", "results-knn.csv", "results-svm.csv")
#resultsTechniques = c("Random Forest", "Logistic Regression", "Decision Tree", "Boosted Gradient", "KNN", "SVM")

resultsFiles = c("results-lower-threshold-svm.csv", "results-lower-threshold-logistic.csv", "results-lower-threshold-forest.csv", "results-lower-threshold-decision.csv", "results-lower-threshold-gradient.csv", "results-lower-threshold-knn.csv")
resultsTechniques = c("SVM", "Random Forest", "Logistic Regression", "Decision Tree", "Boosted Gradient", "KNN")

results <- data.frame(resultsFiles, resultsTechniques)
colnames(results) <- c(RESULT_RESULT, RESULT_TECHNIQUE)

processResults <- function(resultsCSV, technique){

  cropPredictions <- read.csv(resultsCSV)

  totalWeeds = sum(cropPredictions$actual)
  
  # Replace the 0s and 1s with more readable values
  cropPredictions$type = ifelse(cropPredictions$type == 0, "CROP", "WEED")
  cropPredictions$actual = ifelse(cropPredictions$actual == 0, "CROP", "WEED")

  cropPredictions$type <- as.factor(cropPredictions$type)
  cropPredictions$actual <- as.factor(cropPredictions$actual)
  confusionMatrix <- conf_mat(cropPredictions, truth=actual, estimate=type)
  
  # TODO: Must be a better way to form a path to a file
  filePlot=paste("figures/confusion-",paste(tolower(str_replace(technique, " ", "-")), ".jpg", sep=""), sep="")
  
  plot <- autoplot(confusionMatrix, type="heatmap", ) +
            labs(title=technique) +
            scale_fill_gradient(low="#D6EAF8",high = "#2E86C1") +
            theme(text = element_text(size = 30))    
  ggsave(filename=filePlot, plot=plot)


  # Get the error we care about
  misclassificationOfCropAsWeed = confusionMatrix$table[2,1]
  
  # The number of plants
  totalPopulation = nrow(cropPredictions)
  # The numbers of crop (Weeds are 1s)
  totalCrop = totalPopulation - totalWeeds
  missRate <- (misclassificationOfCropAsWeed / totalCrop) * 100
  
  correctClassificationOfWeed = confusionMatrix$table[2,2]
  hitRate <- (correctClassificationOfWeed / totalWeeds) * 100
  
  summaryOfResults <- summary(confusionMatrix)
  
  # Temporarily, just return the f score:
  #stats <- summary(confusionMatrix)
  #fscore <- f_meas(cropPredictions, truth=actual, estimate = type)
  #fscore$.estimate

  c(technique, missRate, hitRate, summaryOfResults)
}

# Produce the confusion matrices and get the misclassification we care about -- crop as weeds

misclassifications <- vector()
correctClassifications <- vector()
typesMiss <- vector()
typesCorrect <- vector()
techniques <- vector()
summaries <- list()

for (i in 1:nrow(results)){
  list[technique, misclassificationsOfCrop, correctClassificationsOfWeed, summaryOfResults] <- processResults(results[i,1], results[i,2])
  techniques <- append(techniques, technique)

  misclassifications <- append(misclassifications, misclassificationsOfCrop)
  correctClassifications <- append(correctClassifications, correctClassificationsOfWeed)
  summaries <- append(summaries, summaryOfResults)
  # Originally
  #misclassificationsOfCrop <- processResults(results[i,1], results[i,2])
  #misclassifications <- append(misclassifications, misclassificationsOfCrop)
}
misclassificationsOfCropAsWeeds <- data.frame(matrix(ncol=3, nrow=length(resultsTechniques)))
colnames(misclassificationsOfCropAsWeeds) <- c('Technique', 'Misclassification', 'CorrectClassifications')

misclassificationsOfCropAsWeeds$Technique = results$technique
misclassificationsOfCropAsWeeds$Misclassification = misclassifications
misclassificationsOfCropAsWeeds$CorrectClassifications = correctClassifications

#misclassificationsOfCropAsWeeds$Type = c(typesMiss, typesCorrect)

# ggplot(misclassificationsOfCropAsWeeds, aes(x=reorder(Technique, -Misclassification), y=Misclassification)) +
#   labs(title="Misclassifications of Crop as Weed", x="Technique", y="Misclassification Rate") +
#   geom_bar(stat="identity") +
#   theme(text = element_text(size=20)) +
#   coord_flip()


# Back to back bar chart

library(plotrix)
library(viridis)
library(RColorBrewer)

mcol<-plotrix::color.gradient(c(0,0,0.5,1),c(0,0,0.5,1),c(1,1,0.5,1),6)
fcol<-plotrix::color.gradient(c(1,1,0.5,1),c(0.5,0.5,0.5,1),c(0.5,0.5,0.5,1),6)
# removed labels in center but you could run the example and see another approach
par(family ="serif", cex.main=2.0)
par(mar=plotrix::pyramid.plot(as.numeric(misclassifications),as.numeric(correctClassifications), 
                              labels=misclassificationsOfCropAsWeeds$Technique,
                              top.labels = c("Crops as Weeds", "","Weeds as Weeds"),
                              main="Classification Rates for Weeds/Crops",
                              lxcol=brewer.pal(n = length(misclassificationsOfCropAsWeeds$Technique), name="Dark2"),
                              rxcol=brewer.pal(n = length(misclassificationsOfCropAsWeeds$Technique), name="Dark2"),
                              #rxcol=viridis(length(misclassificationsOfCropAsWeeds$Technique)),
                              xlim=c(1.2,1.2),
                              laxlab = c(0,25),
                              raxlab = c(0,25,50,75,100),
                              ppmar=c(4,3,4,3),
                              gap=15, show.values=TRUE))

