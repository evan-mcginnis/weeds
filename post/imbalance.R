#
# I M B A L A N C E
#

# 
# Produce a graph that shows the change in F1 
library(dplyr)
library(ggplot2)
library(gghighlight)
library("GGally")

library(lemon)

setwd("c:/uofa/weeds/post")
imbalances <- read.csv("imbalance.csv")

imbalances$classification <- as.factor(imbalances$classification)
imbalances$correction <- as.factor(imbalances$correction)
imbalances$ratio <- as.factor(imbalances$ratio)

#
# This is a version that puts the ratio at tbe base of each bar
# Too busy, perhaps
#

# Lollipop
ggplot(imbalances, aes(x = classification, 
                      y = f1.delta,
                      fill = as.factor(ratio))) +
  # For a lollipop chart -- next two lines
  geom_segment(position=position_dodge(0.95), aes(x = classification, xend = classification, y = 0, yend = f1.delta)) +
  geom_point() +
  xlab("Classification Method") +
  ylab("Delta on F1") +
  facet_wrap(~correction) +
  #geom_col(position = "dodge", show.legend = T) +
  #  labs(fill="Ratio") +
  #geom_text(aes(label = format(round(f1.delta, digits=2), nsmall=2), y = 0),
  #          position = position_dodge(width=0.9),
  #          hjust = 1,
  #          angle = -90,
  #          size = 2) +
  theme(axis.text.x = element_text(angle = -90, hjust = 0))

#
# This is the version without the ratio at the bottom of the bar
#
ggplot(imbalances, aes(x = classification,
                       y = auc.delta,
                       fill = as.factor(ratio),
                       width=1.0)) +
  xlab("Classification Method") +
  ylab("Delta of AUC") +
  facet_wrap(~correction) +
  coord_cartesian(ylim = c(-0.05,0.2)) +
  #
  annotate("rect", xmin = -Inf, xmax = Inf, ymin = -Inf, ymax = 0, fill = "grey", alpha = 0.30) +
  #
  geom_col(position = position_dodge(0.95), show.legend = T) +
  labs(fill="Ratio") +
  #geom_text(aes(label = ratio, y = -.095),
  #          position = position_dodge(width=0.9),
  #          hjust = 1,
  #          angle = -90,
  #          size = 2) +
  theme(axis.text.x = element_text(angle = -90, hjust = 0.0, vjust = 0.3))

#
# Gillian's version

imbalances |>
  mutate(ratio = as.factor(ratio)) |>
  ggplot(aes(x = classification,
             y = f1.delta,
             group = ratio,
             fill = ratio,
             color = ratio)) +
  facet_wrap(~correction) +
  geom_col(position = position_dodge(width = 0.9), width = 0.1) +
  geom_point(position = position_dodge(width = 0.9)) +
  geom_hline(yintercept = 0, color = "black") +
  theme(axis.text.x = element_text(angle = -90, hjust = 0.0, vjust = 0.3))

