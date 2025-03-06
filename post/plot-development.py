import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2 as cv
import sys
import os

import constants

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss

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

cropObservations = unstacked.iloc[0].to_numpy()
weedObservations = unstacked.iloc[1].to_numpy()

adf = adfuller(cropObservations, autolag='t-stat')
print(f"Crop {cropObservations} ADF: {adf}")
adf = adfuller(weedObservations, autolag='t-stat')
print(f"Weed {weedObservations} ADF: {adf}")

plt.style.use('ggplot')
# Plot the means
unstacked.plot.barh(title=arguments.title)
ax = plt.gca()
xAxisNames = ["Weed", "Crop"]
ax.set_yticklabels(xAxisNames, rotation='vertical')
plt.ylabel("Type")
plt.xlabel(arguments.factor)
# Form the legend from the index names -- the dates of the observations
dates = list(dfGroupedByDate.index)
formatted_dates = [date.strftime('%Y-%m-%d') for date in dates]
print(f"Dates: {formatted_dates}")
plt.legend(formatted_dates, bbox_to_anchor=(1.04, 0.5), loc='center left')
#ax.legend(list(dfGroupedByDate.index), loc='upper left')

if arguments.output is not None:
    plt.savefig(arguments.output, bbox_inches="tight")
else:
    plt.show()

sys.exit(0)