#
# S E G M E N T A T I O N  E R R O R S
#
#
library(ggplot2)
library(GGally)
library(gt)
library("dplyr")
library(tidyverse)

setwd("c:/uofa/weeds/util")

input <- "results.csv"
#input <- "results-latest.csv"
output <- "../documents/figures"

# Get the file name from the command line
#cropPredictions <- read.csv(opt$features)
segmentation <- read.csv(input)

 
means <- segmentation %>%
  group_by(technique) %>%
  summarise_at(c("I", "II", "Total"), mean, na.rm = TRUE)
means <- arrange(means, Total)
keeps <- c("Technique", "I", "II")
means <- means[keeps]

ggplot(means) +
  geom_bar(aes(x=technique, y=I), stat="identity") +
  coord_flip()

means$technique <- as.factor(means$technique)

x <- means %>%
  pivot_longer(cols = c("I", "II"))

# g's code
means %>%
  arrange(Total) %>%
  mutate(technique = factor(technique, levels = technique)) %>%
  pivot_longer(cols = c("I", "II")) %>%
  ggplot(aes(x = value, y = technique, fill = name)) +
  geom_col() +
  xlab("Error rate") +
  ylab("Technique") +
  labs(fill="Type", title = "Error Rates of Segmentation Techniques")
