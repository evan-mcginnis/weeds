# 
# F I N A L
# 
# Evan McGinnis
#
#
library(raster)
library(lidR)


# Set the working directory
#setwd("s:\\422evanmc\\final-project\\final\\2_densification\\point_cloud")
setwd("c:\\University of Arizona\\RNR522\\weeds")
# The macos variant
#setwd("~/Box/RNR-522/final/final")
# read in the cloud data
#testcloud = readLAS("2_densification/point_cloud/final_group1_densified_point_cloud.las")
testcloud = readLAS("final_group1_densified_point_cloud.las")
testcloud = readLAS("manipulated.las")
plot(testcloud, color="RGB")
vox <- voxelize_points(testcloud,.5)
plot(vox, voxel = T, bg="white")

canopyVoxel = lasvoxelize(testcloud, res = 0.10)
vox_met <- voxel_metrics(canopyVoxel, ~list(N = length(Z)), .5)
plot(vox_met, color="N", colorPalette = heat.colors(50), size = .5, bg = "white", voxel = T)
plot(canopyVoxel)

voxelsLAD = voxel_metrics(testcloud, ~LAD(z=las@data$Z, dz = 1, k= 0.5, z0= 1), 5)
plot(voxelsLAD, color = "lad", colorPalette = heat.colors(5), bg = "black", legend = TRUE)

# lasground is deprecated.  TBD to replace this with classify_ground

classified = lasground(testcloud, algorithm = csf(sloop_smooth = FALSE, class_threshold = 0.1, cloth_resolution = 0.5, rigidness = 2))

plot(classified, color="Classification")

ground_points = lasfilter(classified, Classification == 2)
plot(ground_points, color = "RGB")

canopy_points = lasfilter(classified, Classification == 0)
plot(canopy_points, color = "RGB")

classified_experiment = lasground(testcloud, algorithm = csf(sloop_smooth = FALSE, class_threshold = 1, cloth_resolution = 0.5, rigidness = 2))
ground_points_e = lasfilter(classified_experiment, Classification == 2)
plot(ground_points_e, color = "RGB")

canopy_points_e = lasfilter(classified_experiment, Classification == 0)
plot(canopy_points_e, color = "RGB")

# Step 22.  Looks like it uses KNN
ground_dtm = grid_terrain(ground_points, res=0.1, algorithm = knnidw(k =10, p=2))
# Step 23
plot(ground_dtm, xlab="UTM_X", ylab= "UTM_Y", main="Ground_DTM")
mtext("Elevation (meter)", side=4)

# Step 24
writeRaster(ground_dtm, filename="ground_dtm.tif", format="GTiff", datatype='FLT4S', overwrite=TRUE)

# Step 25
agl = lasnormalize(canopy_points, ground_dtm, copy=TRUE)

# Step 26
plot(agl)

# Step 27
range(agl $Z)

# Step 28
aglClean = lasfilter(agl, Z > 0)
range(aglClean $Z)

# Step 30
canopyVoxel = lasvoxelize(aglClean, res = 0.10)

# Step 33
treeseg4 = lastrees(canopyVoxel, li2012(dt1 = 1.5, dt2 = 2, R = 2, Zu = 10, hmin=1.5, speed_up = 5), attribute = "treeID" )

# Step 34
plot(treeseg4, color="treeID")

# Step 35
uniqueTrees = unique(treeseg4$treeID)
length(uniqueTrees)
