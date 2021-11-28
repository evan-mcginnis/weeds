import sys

import numpy as np
import laspy
#from laspy.file import File

#with laspy.open('./final_group1_densified_point_cloud.las') as fh:
#    print('Points from Header:', fh.header.point_count)
#    las = fh.read()
las = laspy.read('./final_group1_densified_point_cloud.las')
print(las)
print('Points from data:', len(las.points))
points = las.points
format = las.point_format
print('Format:', format.id)
rPoints = las.red
gPoints = las.green
bPoints = las.blue
las.red = np.where(las.red > 0,0,0)
#for point in rPoints:
#    point = 0
#las.red = rPoints
for point in las.green:
    point = 0
las.green = gPoints
for point in bPoints:
    point = 0
las.blue = bPoints
las.write("manipulated.las")
    #print("RGB: {}".format(point))
#ground_pts = las.classification == 2
#bins, counts = np.unique(las.return_number[ground_pts], return_counts=True)
#print('Ground Point Return Number distribution:')
#for r,c in zip(bins,counts):
#    print('    {}:{}'.format(r,c))

sys.exit(0)

# Read in LAS file
las = laspy.read()
# Extract the color data as numpy array
colorData = extractColorData(las)
# Create an index
index =
# Get 


