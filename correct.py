from VegetationIndex import VegetationIndex
import numpy as np
import argparse

veg = VegetationIndex()

parser = argparse.ArgumentParser("Correct an image with a specified vegetation index.")

parser.add_argument('-i', '--input', action="store", required=True, help="Input image")
parser.add_argument('-o', '--output', action="store", required=True, help="Output image")
parser.add_argument("-a", '--algorithm', action="store", required=True, help="Vegetation Index algorithm",
                    choices=veg.GetSupportedAlgorithms(),
                    default="com1")
parser.add_argument("-t", "--threshold", action="store", required=True, type=int)
parser.add_argument("-v", "--verbose", action="store_true")

results = parser.parse_args()

veg.Load(results.input)
image = veg.GetImage()
if results.verbose:
    veg.ShowImage("Source", image)
index = veg.Index(results.algorithm)

if results.verbose:
    veg.ShowImage("Index", index)

veg.MaskFromIndex(index, results.threshold)
veg.applyMask()
image = veg.GetMaskedImage()

if results.verbose:
    veg.ShowImage(results.algorithm + " mask applied to source", image)

veg.SaveIndexToFile(results.output, image)
