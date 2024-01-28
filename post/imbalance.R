#
# I M B A L A N C E
#

library(dplyr)
library(ggplot2)
library(gghighlight)
library("GGally")

library(lemon)

imbalances <- read.csv("imbalance.csv")

imbalances$classification <- as.factor(imbalances$classification)
imbalances$correction <- as.factor(imbalances$correction)
imbalances$minority <- as.factor(imbalances$minority)



ggplot(imbalances, aes(x = classification,
                      y = auc,
                      fill = as.factor(minority))) +
  xlab("Classification Method") +
  ylab("Area under the curve") +
  facet_wrap(~correction) +
  geom_col(position = "dodge", show.legend = T) +
    labs(fill="Minority Class") +
  geom_text(aes(label = format(round(auc, digits=2), nsmall=2), y = 0),
            position = position_dodge(width=0.9),
            hjust = 1,
            angle = -90,
            size = 2) +
  theme(axis.text.x = element_text(angle = -90, hjust = 0))

p<-ggplot(df, aes(x=Category, y=Mean, fill=Quality)) +  
  geom_point()+ 
  geom_errorbar(aes(ymin=Mean-sd, ymax=Mean+sd), width=.2, 
                position=position_dodge(0.05)) 

p

