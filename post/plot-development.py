import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2 as cv
import sys
import os

import constants

parser = argparse.ArgumentParser("Plot factor over development cycle")
parser.add_argument("-i", "--input", action="store", required=True, nargs="*", help="Image to process")
parser.add_argument("-f", "--factor", action="store", required=True, help="Name of factor")
parser.add_argument("-t", "--title", action="store", required=False, type=str, help="Title of plot")
parser.add_argument("-o", "--output", action="store", required=False, help="Output graph")
arguments = parser.parse_args()

resultsOverCycle = {}
dfAll = pd.DataFrame()

for datafile in arguments.input:
    print(f"Reading: {datafile}")
    if not os.path.isfile(datafile):
        print(f"Unable to access: {datafile}")
        sys.exit(-1)

    # Read in datafile as a dataframe
    df = pd.read_csv(datafile)

    # See if the factor is there
    if arguments.factor not in df.columns:
        print(f"Factor not found: {arguments.factor}")
        sys.exit(-1)
    if constants.NAME_DATE not in df.columns:
        print(f"Unable to file acquisition date")
        sys.exit(-1)

    dfAll = dfAll.append(df)

# Compute the mean where there are several dates in the result file
dfGrouped = dfAll.groupby(constants.NAME_DATE).agg(factor=(arguments.factor, "mean"))
print(f"{dfGrouped}")


print(f"Mean of {arguments.factor}: {df[arguments.factor].mean()}")
