#
# F A C T O R  P L O T
#

library(rgl)
library(ConfigParser)

library(optparse)


option_list = list(
  make_option(c("-i", "--ini"), type="character", default="standalone.ini", 
              help="INI file name", metavar="character"),
  make_option(c("-f", "--features"), type="character", default="results-latest.csv", 
              help="Features extracted [default= %default]", metavar="character")
); 

opt_parser = OptionParser(option_list=option_list);
opt = parse_args(opt_parser);

setwd("c:/uofa/weeds/jetson")

input <- "results-latest.csv"
#input <- "results-latest.csv"
output <- "figures"

# Get the file name from the command line
#cropPredictions <- read.csv(opt$features)
cropPredictions <- read.csv("dissertation.csv")

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

classificationFactorsString <- config$get(factorsKey, NA, factorsSection)
classificationFactors <- unlist(strsplit(classificationFactorsString, split=", "))

if (is.na(classificationFactorsString)) {
  err <- sprintf("Unable to find factors specified by %s in %s", factorsKey, iniFile)
  stop(err)
}

# Gillian's testing zone
library(dplyr)
library(scales)

excludedVars <- c("name", "number", "type", "actual", "date")

# Rescale is hidden, so specify explicitly
scaledCropPreds <- cropPredictions %>%
  mutate_at(
    vars(-(excludedVars)),
    ~(scales::rescale(., to=c(0,1)))
  )
# end of gillian's testing zone



# The RGL method
# plot3d(
#   x=cropPredictions$lw_ratio,
#   y=cropPredictions$shape_index,
#   z=cropPredictions$in_phase,
#   type='s',
#   radius=.1,
#   xlab="LW Ratio", ylab="Shape Index", zlab="YIQ")

library(plotly)
library(dbplyr)

colorBG <- "rgb(192, 192, 192)"
colorGrid <- "rgb(211,211,211)"
colorLine <- "rgb(255,255,255)"

axx <- list(
  title = classificationFactors[1],
  backgroundcolor=colorBG,
  gridcolor=colorLine,
  showbackground=TRUE,
  zerolinecolor=colorLine
)

axy <- list(
  title = classificationFactors[2],
  backgroundcolor=colorBG,
  gridcolor=colorLine,
  showbackground=TRUE,
  zerolinecolor=colorLine
)

axz <- list(
  title = classificationFactors[3],
  backgroundcolor=colorBG,
  gridcolor=colorLine,
  showbackground=TRUE,
  zerolinecolor=colorLine
)

#cropPredictions <- scaledCropPreds
cropPredictions$cluster = as.factor(cropPredictions$actual)
# Works
#p <- plot_ly(cropPredictions, x=~lw_ratio, y=~shape_index, z=~in_phase, color=~normalized_distance, symbol=~I(actual), hovertext=cropPredictions$name, hoverinfo="x+y+z+text") 

# Feature Importance
#p <- plot_ly(cropPredictions, x=~contrast, y=~saturation_mean, z=~in_phase, color=~dissimilarity, symbol=~I(actual), hovertext=cropPredictions$name, hoverinfo="x+y+z+text") 

# Recursive
#axx['title'] <- "Saturation Mean"
#axy['title'] <- "YIQ In-Phase"
#axz['title'] <- "Dissimilarty"
#techique <- "Recursive"
#p <- plot_ly(cropPredictions, x=~saturation_mean, y=~in_phase, z=~dissimilarity, color=~ASM, symbol=~I(actual), hovertext=cropPredictions$name, hoverinfo="x+y+z+text")

# Variance
axx['title'] <- classificationFactors[1]
axy['title'] <- classificationFactors[2]
axz['title'] <- classificationFactors[3]
technique <- "LDA"
p <- plot_ly(cropPredictions, x=~eval(as.symbol(classificationFactors[1])), y=~eval(as.symbol(classificationFactors[2])), z=~eval(as.symbol(classificationFactors[3])), color=~eval(as.symbol(classificationFactors[4])), symbol=~I(actual), hovertext=cropPredictions$name, hoverinfo="x+y+z+text")

# PCA
#axx['title'] <- "Hue"
#axy['title'] <- "Saturation Mean"
#axz['title'] <- "YIQ In-Phase"
#technique <- "PCA"
#p <- plot_ly(cropPredictions, x=~hue, y=~saturation_mean, z=~in_phase, color=~cb_mean, symbol=~I(actual), hovertext=cropPredictions$name, hoverinfo="x+y+z+text")

# Univariate
#axx['title'] <- "Hue"
#axy['title'] <- "Saturation Mean"
#axz['title'] <- "YIQ In-Phase"
#p <- plot_ly(cropPredictions, x=~hue, y=~saturation_mean, z=~in_phase, color=~dissimilarity, symbol=~I(actual), hovertext=cropPredictions$name, hoverinfo="x+y+z+text")
#technique <- "Univariate"
#p <- p %>% colorbar(title="Dissimilarity")

p <- p %>% add_markers()
# As I can't have a second legend in plotly, I will just put the shape interpretation in the title
plotTitle <- sprintf("%s: Features of Weeds [Circles] and Crop [Squares] Crop as Weed [Diamonds]", technique)
# This works, but I get an ugly title for the color bar
# p <- p %>% layout(scene = list(xaxis=axx,yaxis=axy,zaxis=axz), title=plotTitle )
p <- p %>% layout(scene = list(xaxis=axx,yaxis=axy,zaxis=axz), title=plotTitle)
p <- p %>% colorbar(title = classificationFactors[4])
#p <- p %>% layout(legend = list(x = 100, y = 0.5, title=list(text=' TEXT ')))
#p <- p %>% colorbar(title="Normalized distance from cropline")
#p <- p %>% colorbar(title="ASM")
#p <- p %>% add_trace(y = "blah")

p



