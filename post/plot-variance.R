#
# F A C T O R  P L O T
#

library(rgl)
library(ConfigParser)

library(optparse)
library(ggplot2)


option_list = list(
  make_option(c("-i", "--ini"), type="character", default="standalone.ini", 
              help="INI file name", metavar="character"),
  make_option(c("-f", "--features"), type="character", default="results-latest.csv", 
              help="Features extracted [default= %default]", metavar="character")
); 

opt_parser = OptionParser(option_list=option_list);
opt = parse_args(opt_parser);

setwd("c:/uofa/weeds/jetson")

input <- "normalized.csv"
#input <- "results-latest.csv"
output <- "figures"

# Get the file name from the command line
#cropPredictions <- read.csv(opt$features)
cropPredictions <- read.csv("normalized.csv")

#
# I N I  P R O C E S S I N G
#
# Read the factors used from the INI file

iniFile <- "standalone.ini"
factorsKey <- "FACTORS_R"
factorsSection <- "IMAGE-PROCESSING"

config <- ConfigParser$new()

# Get the file name from the command line
#config$read(opt$ini)
config$read(iniFile)

#classificationFactorsString <- config$get("FACTORS_R_PCA", NA, factorsSection)
#classificationFactors <- unlist(strsplit(classificationFactorsString, split=", "))

getFactors <- function(config, factorsKey, factorsSection)
{
  classificationFactorsString <- config$get(factorsKey, NA, factorsSection)
  classificationFactors <- unlist(strsplit(classificationFactorsString, split=", "))
  
  if (is.na(classificationFactorsString)) {
    err <- sprintf("Unable to find factors specified by %s in %s", factorsKey, iniFile)
    stop(err)
  }
  return (classificationFactors)
}

factorsPCA <- getFactors(config, "FACTORS_R_PCA", factorsSection)
factorsImportance <- getFactors(config, "FACTORS_R_IMPORTANCE", factorsSection)
factorsUnivariate <- getFactors(config, "FACTORS_R_UNIVARIATE", factorsSection)
factorsRecursive <- getFactors(config, "FACTORS_R_RECURSIVE", factorsSection)

factorsAll <- c(factorsPCA, factorsImportance, factorsUnivariate, factorsRecursive)

#factorsUnique <- unique(factorsAll)
# Don't make the list unique so we can put in which factor applies
factorsUnique <- factorsAll

techniqueNames <- c("PCA", "Importance", "Univariate", "Recursive")
techniques <- data.frame(name=rep(techniqueNames, times=c(10,10,10,10)))
  

variances = list()
for (parameter in factorsUnique) {
  #print(parameter)
  #print(var(cropPredictions[parameter]))
  variances[length(variances)+1] = var(cropPredictions[parameter])
}
variances <- unlist(variances)


variancesByParameter <- data.frame(name=factorsUnique, variance=variances, technique=techniques$name, stringsAsFactors = FALSE)

subset <- cropPredictions %>% select(classificationFactors)

variancesByParameter %>% 
  mutate(rownum = row_number()) %>% 
  left_join(
    techniques %>% mutate(rownum = row_number()) %>% rename(technique = name),
    by = join_by(rownum)
  )
  
subset %>% 
  summarise(mean = var(hue)) %>% 
  ggplot(aes(x = hue, y = mean)) + 
  geom_col()

p <-ggplot(data=variancesByParameter, aes(x=variance, y = name)) +
  ylab("Higher Rank >") +
  xlab("Variance of Normalized Feature") +
  geom_bar(color="blue", fill="blue" ,stat="identity") + 
  facet_wrap(. ~ as.factor(technique), scales='free_y') +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))
p


