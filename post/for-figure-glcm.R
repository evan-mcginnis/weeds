#
# GLCM Comparison
#
#
library(ggplot2)
library(GGally)
library(gt)

setwd("c:/uofa/weeds/jetson")

input <- "corrected.csv"
#input <- "results-latest.csv"
output <- "figures"

# Get the file name from the command line
#cropPredictions <- read.csv(opt$features)
cropPredictions <- read.csv(input)

# GLCM
pdf("glcm-pairs.pdf", height = 7, width = 7)
p <- ggpairs(cropPredictions, 
             aes(col = as.character(type)),
             title = "GLCM Values",
             lower = list(alpha = 0.1),
             columns = c("greyscale_homogeneity_avg",
                         "greyscale_energy_avg", 
                         "greyscale_ASM_avg", 
                         "greyscale_contrast_avg", 
                         "greyscale_dissimilarity_avg", 
                         "greyscale_correlation_avg"),
             columnLabels = c("Homogeneity",
                              "Energy",
                              "ASM",
                              "Contrast",
                              "Dissimilarity",
                              "Correlation"))

print(p)
dev.off()

# HOG
pdf("hog-pairs.pdf", height = 7, width = 7)
p <- ggpairs(cropPredictions, 
             aes(col = as.character(type)), 
             title="HOG Values",
             columns = c("hog_stddev",
                         "hog_variance",
                         "hog_mean"),
             columnLabels = c("Standard Deviation",
                              "Variance",
                              "Mean"),
             diag = list(continuous = "blankDiag"))

print(p)
dev.off()

bones.d <- bones %>% select(-Species)
nor <- scale(bones.d,center=means,scale=sds)

k2 <- kmeans(nor, centers = 4, nstart = 5)

bones.pca <- prcomp(bones.d, scale = T)

bones.d %>% 
  scale() %>% 
  prcomp() %>%  
  .$rotation

plot1 <- fviz_cluster(k2, 
                      data = nor, 
                      show.clust.cent = F,
                      axes=c(1,2),
                      #shape = c(1,2,3,4),
                      main="Cervidae Clusters",
                      geom = "point") +
  theme_minimal() +
  theme(text = element_text(size=14),  plot.tag = element_text(size=28)) +
  xlab("PCA 1") +
  ylab("PCA 2") +
  labs(tag = "A", element_text(size=12))


