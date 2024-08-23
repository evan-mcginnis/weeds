import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import cv2 as cv


METHOD = 'uniform'
plt.rcParams['font.size'] = 9

parser = argparse.ArgumentParser("Plot LBP for two images")

parser.add_argument('-l', '--left', action="store", required=True, type=str, help="left image")
parser.add_argument('-r', '--right', action="store", required=True, type=str, help="right image")
parser.add_argument('-o', '--output', action="store", help="Output directory for graph")

arguments = parser.parse_args()

# Make sure the images are there, read them, and convert to greyscale
if not os.path.isfile(arguments.left):
    print(f"Unable to access: {arguments.left}")
    exit(-1)
else:
    img = cv.imread(arguments.left)
    left = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
if not os.path.isfile(arguments.right):
    print(f"Unable to access: {arguments.right}")
    exit(-1)
else:
    img = cv.imread(arguments.right)
    right = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

if not os.path.isdir(arguments.output):
    print(f"Unable to access output directory: {arguments.output}")
    exit(-1)


def plot_circle(ax, center, radius, color):
    circle = plt.Circle(center, radius, facecolor=color, edgecolor='0.5')
    ax.add_patch(circle)


def plot_lbp_model(ax, binary_values):
    """Draw the schematic for a local binary pattern."""
    # Geometry spec
    theta = np.deg2rad(45)
    R = 1
    r = 0.15
    w = 1.5
    gray = '0.5'

    # Draw the central pixel.
    plot_circle(ax, (0, 0), radius=r, color=gray)
    # Draw the surrounding pixels.
    for i, facecolor in enumerate(binary_values):
        x = R * np.cos(i * theta)
        y = R * np.sin(i * theta)
        plot_circle(ax, (x, y), radius=r, color=str(facecolor))

    # Draw the pixel grid.
    for x in np.linspace(-w, w, 4):
        ax.axvline(x, color=gray)
        ax.axhline(x, color=gray)

    # Tweak the layout.
    ax.axis('image')
    ax.axis('off')
    size = w + 0.2
    ax.set_xlim(-size, size)
    ax.set_ylim(-size, size)


#fig, axes = plt.subplots(ncols=5, figsize=(7, 2))

titles = ['flat', 'flat', 'edge', 'corner', 'non-uniform']

binary_patterns = [
    np.zeros(8),
    np.ones(8),
    np.hstack([np.ones(4), np.zeros(4)]),
    np.hstack([np.zeros(3), np.ones(5)]),
    [1, 0, 0, 1, 1, 1, 0, 0],
]

# for ax, values, name in zip(axes, binary_patterns, titles):
#     plot_lbp_model(ax, values)
#     ax.set_title(name)

from skimage.transform import rotate
from skimage.feature import local_binary_pattern
from skimage import data
from skimage.color import label2rgb

# settings for LBP
radius = 3
n_points = 8 * radius


def overlay_labels(image, lbp, labels):
    mask = np.logical_or.reduce([lbp == each for each in labels])
    return label2rgb(mask, image=image, bg_label=0, alpha=0.5)


def highlight_bars(bars, indexes):
    for i in indexes:
        bars[i].set_facecolor('r')


image = data.brick()
lbp = local_binary_pattern(image, n_points, radius, METHOD)


def hist(ax, lbp):
    n_bins = int(lbp.max() + 1)
    ignored = 0
    # Originally
    #return ax.hist(lbp.ravel(), density=True, bins=n_bins - ignored, range=(0, n_bins - ignored), facecolor='0.5')
    return ax.hist(lbp.ravel(), density=False, log=True, bins=n_bins - ignored, range=(0, n_bins - ignored), facecolor='black')

def histBrokenY(ax, lbp):
    n_bins = int(lbp.max() + 1)
    ignored = 0
    myHistogram = np.histogram(lbp.ravel(), density=True, bins=n_bins - ignored, range=(0, n_bins - ignored), facecolor='0.5')
    f, (ax, ax2) = plt.subplot(2, 1, sharex=True, facecolor='w')  # make the axes
    # ax.bar(bin_centres, my_hist)  # plot on top axes
    # ax2.bar(bin_centres, my_hist)  # plot on bottom axes
    ax.set_ylim([0.0, 0.14])  # numbers here are specific to this example
    ax2.set_ylim([0.14, 0.5])  # numbers here are specific to this example
    ax.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax.xaxis.tick_bottom()
    ax.tick_params(labeltop=False)
    ax2.xaxis.tick_bottom()

# BEM REMOVE BEGIN
# plot histograms of LBP of textures
# fig, (ax_img, ax_hist) = plt.subplots(nrows=2, ncols=3, figsize=(9, 6))
# plt.gray()
#
# titles = ('edge', 'flat', 'corner')
# w = width = radius - 1
# edge_labels = range(n_points // 2 - w, n_points // 2 + w + 1)
# flat_labels = list(range(0, w + 1)) + list(range(n_points - w, n_points + 2))
# i_14 = n_points // 4  # 1/4th of the histogram
# i_34 = 3 * (n_points // 4)  # 3/4th of the histogram
# corner_labels = list(range(i_14 - w, i_14 + w + 1)) + list(
#     range(i_34 - w, i_34 + w + 1)
# )
#
# label_sets = (edge_labels, flat_labels, corner_labels)
#
# for ax, labels in zip(ax_img, label_sets):
#     ax.imshow(overlay_labels(image, lbp, labels))
#
# for ax, labels, name in zip(ax_hist, label_sets, titles):
#     counts, _, bars = hist(ax, lbp)
#     highlight_bars(bars, labels)
#     ax.set_ylim(top=np.max(counts[:-1]))
#     ax.set_xlim(right=n_points + 2)
#     ax.set_title(name)
#
# ax_hist[0].set_ylabel('Percentage')
# for ax in ax_img:
#     ax.axis('off')
# BEM REMOVE END

# settings for LBP
radius = 4
n_points = 8 * radius


def kullback_leibler_divergence(p, q):
    p = np.asarray(p)
    q = np.asarray(q)
    filt = np.logical_and(p != 0, q != 0)
    return np.sum(p[filt] * np.log2(p[filt] / q[filt]))


def match(refs, img):
    best_score = 10
    best_name = None
    lbp = local_binary_pattern(img, n_points, radius, METHOD)
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp, density=True, bins=n_bins, range=(0, n_bins))
    for name, ref in refs.items():
        ref_hist, _ = np.histogram(ref, density=True, bins=n_bins, range=(0, n_bins))
        score = kullback_leibler_divergence(hist, ref_hist)
        if score < best_score:
            best_score = score
            best_name = name
    return best_name


# brick = data.brick()
# grass = data.grass()
# gravel = data.gravel()

refs = {
    'left': local_binary_pattern(left, n_points, radius, METHOD),
    'right': local_binary_pattern(right, n_points, radius, METHOD),
}

# classify rotated textures
# print('Rotated images matched against references using LBP:')
# print(
#     'original: brick, rotated: 30deg, match result: ',
#     match(refs, rotate(brick, angle=30, resize=False)),
# )
# print(
#     'original: brick, rotated: 70deg, match result: ',
#     match(refs, rotate(brick, angle=70, resize=False)),
# )
# print(
#     'original: grass, rotated: 145deg, match result: ',
#     match(refs, rotate(grass, angle=145, resize=False)),
# )

# plot histograms of LBP of textures
fig, ((ax1), (ax3)) = plt.subplots(nrows=2, ncols=1, figsize=(4, 6))
plt.gray()

ax1.imshow(left)
ax1.axis('off')
hist(ax3, refs['left'])
ax3.set_ylabel('Counts (log scale)')
plt.savefig(os.path.join(arguments.output, "lbp-left.jpg"), bbox_inches='tight')
print(f"Figure written as lbp-left.jpg")

fig, ((ax2), (ax4)) = plt.subplots(nrows=2, ncols=1, figsize=(4, 6))
ax2.imshow(right)
ax2.axis('off')
hist(ax4, refs['right'])
#ax4.set_xlabel('Uniform LBP values')
plt.savefig(os.path.join(arguments.output, "lbp-right.jpg"), bbox_inches='tight')
print(f"Figure written as lbp-right.jpg")
#plt.show()

exit(0)



