import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2 as cv
import sys
import os

import constants

parser = argparse.ArgumentParser("Plot factor over development cycle")
parser.add_argument("-i", "--input", action="store", required=True, nargs="*", help="CSV to process")
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
    # Force the date column to be a date
    df['date'] = pd.to_datetime(df['date'])

    # See if the factor is there
    if arguments.factor not in df.columns:
        print(f"Factor not found: {arguments.factor}")
        sys.exit(-1)
    if constants.NAME_DATE not in df.columns:
        print(f"Unable to file acquisition date")
        sys.exit(-1)

    dfAll = dfAll.append(df)

# Compute the mean where there are several dates in the result file
dfGrouped = dfAll.groupby([constants.NAME_TYPE, constants.NAME_DATE]).agg(factor=(arguments.factor, "mean"))
# We won't do much with this -- this is so we can then get the names for the legend
dfGroupedByDate = dfAll.groupby(constants.NAME_DATE).agg(factor=(arguments.factor, "mean"))


#print(f"{dfGrouped}")

unstacked = dfGrouped.unstack()

plt.style.use('ggplot')
# Plot the means
unstacked.plot.barh(title=arguments.title)
ax = plt.gca()
xAxisNames = ["Weed", "Crop"]
ax.set_yticklabels(xAxisNames, rotation='vertical')
plt.ylabel("Type")
plt.xlabel(arguments.factor)
# Form the legend from the index names -- the dates of the observations
ax.legend(list(dfGroupedByDate.index))

if arguments.output is not None:
    plt.savefig(arguments.output)
else:
    plt.show()

sys.exit(0)