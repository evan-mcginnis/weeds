from scipy import ndimage, misc
import matplotlib.pyplot as plt
import numpy as np
import argparse
from skimage.transform import rotate

parser = argparse.ArgumentParser("Rotate image")
parser.add_argument("-i", "--input", action="store", required=True, help="Depth data file in npy format")
parser.add_argument("-d", "--degrees", action="store", required=True, help="Degrees of rotation")
arguments = parser.parse_args()

depth = None

try:
    depth = np.load(arguments.input)
except FileNotFoundError:
    print("Unable to find the file: {}".format(arguments.input))

#rotated = np.rot90(depth, 1)
rotated = rotate(depth, float(arguments.degrees), preserve_range=True)
plt.imshow(rotated, interpolation='none', vmin=430, vmax=470)
plt.show()

# fig = plt.figure(figsize=(10, 3))
# ax1, ax2, ax3 = fig.subplots(1, 3)
# img = plt.imread(arguments.input)
# img_45 = ndimage.rotate(img, 45, reshape=False)
# full_img_45 = ndimage.rotate(img, 45, reshape=True)
# ax1.imshow(img, cmap='gray')
# ax1.set_axis_off()
# ax2.imshow(img_45, cmap='gray')
# ax2.set_axis_off()
# ax3.imshow(full_img_45, cmap='gray')
# ax3.set_axis_off()
# fig.set_layout_engine('tight')
# plt.show()