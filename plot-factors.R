#
# F A C T O R  P L O T
#

library(rgl)

input <- "results-for-plot.csv"
output <- "figures"

cropPredictions <- read.csv(input)

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
  title = "Length/Width Ratio",
  backgroundcolor=colorBG,
  gridcolor=colorLine,
  showbackground=TRUE,
  zerolinecolor=colorLine
)

axy <- list(
  title = "Shape Index",
  backgroundcolor=colorBG,
  gridcolor=colorLine,
  showbackground=TRUE,
  zerolinecolor=colorLine
)

axz <- list(
  title = "YIQ In-Phase Mean",
  backgroundcolor=colorBG,
  gridcolor=colorLine,
  showbackground=TRUE,
  zerolinecolor=colorLine
)

cropPredictions$cluster = as.factor(cropPredictions$actual)
# Works
#p <- plot_ly(cropPredictions, x=~lw_ratio, y=~shape_index, z=~in_phase, color=~normalized_distance, symbol=~I(actual), hovertext=cropPredictions$name, hoverinfo="x+y+z+text") 
p <- plot_ly(cropPredictions, x=~lw_ratio, y=~shape_index, z=~in_phase, color=~normalized_distance, symbol=~I(actual), hovertext=cropPredictions$name, hoverinfo="x+y+z+text") 
p <- p %>% add_markers()
# As I can't have a second legend in plotly, I will just put the shape interpretation in the title
p <- p %>% layout(scene = list(xaxis=axx,yaxis=axy,zaxis=axz), title="Features of Weeds (Circles) and Crop (Squares) Crop as Weed (Diamonds)" )
p <- p %>% layout(legend = list(x = 100, y = 0.5))
p <- p %>% colorbar(title="Normalized distance from cropline")
#p <- p %>% add_trace(y = "blah")

p

