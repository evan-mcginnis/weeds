import argparse
import cv2

parser = argparse.ArgumentParser("Lighten image")
parser.add_argument("-i", "--input", required=True, help="Input image")
parser.add_argument("-o", "--output", required=True, help="Output image")
parser.add_argument("-a", "--alpha", required=False, default=1.5, type=float, help="Alpha (contrast)")
parser.add_argument("-b", "--beta", required=False, default=5, type=float, help="Beta (brightness)")

arguments = parser.parse_args()

image = cv2.imread(arguments.input)

# define the alpha and beta
alpha = arguments.alpha # Contrast control
beta = arguments.beta # Brightness control

# call convertScaleAbs function
adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

cv2.imwrite(arguments.output, adjusted)
# display the output image
# cv2.imshow('adjusted', adjusted)
# cv2.waitKey()
# cv2.destroyAllWindows()