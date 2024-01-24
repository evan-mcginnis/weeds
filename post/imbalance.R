#
# I M B A L A N C E
#

library(dplyr)
library(ggplot2)
library(gghighlight)
library("GGally")

imbalances <- read.csv("imbalance.csv")

imbalances$classification <- as.factor(imbalances$classification)
imbalances$correction <- as.factor(imbalances$correction)
imbalances$minority <- as.factor(imbalances$minority)



ggplot(imbalance, aes(x = classification,
                      y = accuracy,
                      fill = as.factor(minority))) +
  xlab("Classification Method") +
  ylab("Overall accuracy") +
  facet_wrap(~correction) +
  geom_col(position = "dodge", show.legend = F) +
  geom_text(aes(label = format(round(accuracy, digits=2), nsmall=2), y = 0),
            position = position_dodge(width=0.9),
            hjust = 1,
            angle = -90,
            size = 2) +
  theme(axis.text.x = element_text(angle = -90, hjust = 0))
