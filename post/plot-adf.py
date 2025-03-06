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
from tqdm import tqdm

from Factors import Factors
from Factors import FactorTypes
from Factors import FactorSubtypes
from Factors import FactorKind

factorChoices = [e.name for e in FactorTypes]
factorChoices.append(constants.NAME_ALL)
factorSubtypeChoices = [e.name for e in FactorSubtypes]
factorSubtypeChoices.append(constants.NAME_ALL)
factorTypes = [e for e in FactorTypes]
factorSubtypes = [e for e in FactorSubtypes]

parser = argparse.ArgumentParser("Plot ADF over development cycle")
parser.add_argument("-i", "--input", action="store", required=True, nargs="*", help="CSV to process")
#parser.add_argument("-f", "--factor", action="store", required=True, help="Name of factor")
parser.add_argument("-l", "--label", action="store", required=False, type=str, help="Label for table")
parser.add_argument("-c", "--caption", action="store", required=False, type=str, help="Long caption")
parser.add_argument("-cs", "--caption-short", action="store", required=False, type=str, help="Short caption")
parser.add_argument("-s", "--subtype", action="store", required=False, type=str, choices=factorSubtypeChoices, help="Subtype")
parser.add_argument("-t", "--type", action="store", required=False, nargs="*", type=str, choices=factorChoices, help="Type")
parser.add_argument("-thresh", "--threshold", action="store", required=False, default=10, type=int, help="Threshold for graph instead of table")
parser.add_argument("-o", "--output", action="store", required=False, help="Output graph or table")

arguments = parser.parse_args()

if arguments.output is not None:
    if os.path.isfile(arguments.output):
        print(f"Output file {arguments.output} exists. Will not overwrite")
        sys.exit(-1)

if constants.NAME_ALL in arguments.type:
    types = factorTypes
else:
    types = [FactorTypes[e] for e in arguments.type]

if constants.NAME_ALL in arguments.subtype:
    subtypes = factorSubtypes
else:
    subtypes = [FactorSubtypes[e] for e in arguments.subtype]

resultsOverCycle = {}
dfAll = pd.DataFrame()

# Get a subset of factors
possibleFactors = Factors()
allFactorsSubset = possibleFactors.getColumns(types, subtypes, FactorKind.SCALAR)

# Columns that can be ignored
nonFactorColumns = ['name', 'number', 'type', 'agl', 'date']

# The columns we want to read in
columnsToRead = allFactorsSubset.copy()
columnsToRead.extend(nonFactorColumns)



for datafile in arguments.input:
    print(f"Reading: {datafile}")
    if not os.path.isfile(datafile):
        print(f"Unable to access: {datafile}")
        sys.exit(-1)

    # Read in datafile as a dataframe
    #df = pd.read_csv(datafile, index_col=0, usecols=columnsToRead)
    df = pd.read_csv(datafile, usecols=columnsToRead)
    # Force the date column to be a date
    df['date'] = pd.to_datetime(df['date'])

    # See if the factor is there
    # if arguments.factor not in df.columns:
    #     print(f"Factor not found: {arguments.factor}")
    #     sys.exit(-1)
    if constants.NAME_DATE not in df.columns:
        print(f"Unable to file acquisition date")
        sys.exit(-1)

    dfAll = dfAll.append(df)

# # Get a subset of factors
# possibleFactors = Factors()
# allFactorsSubset = possibleFactors.getColumns(types, subtypes, FactorKind.SCALAR)
#
# # Columns that can be ignored
# nonFactorColumns = ['name', 'number', 'type', 'agl', 'date']
# All columns in frame
allFactors = dfAll.columns.tolist()
#factorColumns = [item for item in allFactors if item not in nonFactorColumns]
factorColumns = [item for item in allFactors if item in allFactorsSubset]


# Create a frame for the results
minimalColumns = factorColumns
minimalColumns.append(constants.NAME_TYPE)
adf = {}
for factor in factorColumns:
    adf[factor] = 0

dfADF = pd.DataFrame(columns=minimalColumns)
adf[constants.NAME_TYPE] = 0
dfADF = dfADF.append(adf, ignore_index=True)
adf[constants.NAME_TYPE] = 1
dfADF = dfADF.append(adf, ignore_index=True)
adf[constants.NAME_TYPE] = 3
dfADF = dfADF.append(adf, ignore_index=True)

# Clean up the data a bit.
# Some cells will have 0s in them, so replace with NAN and drop those
for column in allFactorsSubset:
    dfAll[column] = dfAll[column].replace(0, np.nan)
dfAll.dropna(subset=factorColumns, inplace=True)

idx = range(0, len(dfAll))
indices = list(idx)
dfAll['number'] = indices
dfAll.set_index('number')

for i in tqdm(range(len(factorColumns))):
    factor = factorColumns[i]
    # Compute the mean where there are several dates in the result file
    dfGrouped = dfAll.groupby([constants.NAME_TYPE, constants.NAME_DATE]).agg(factor=(factor, "mean"))
    # We won't do much with this -- this is so we can then get the names for the legend
    dfGroupedByDate = dfAll.groupby(constants.NAME_DATE).agg(factor=(factor, "mean"))


    #print(f"{dfGrouped}")

    unstacked = dfGrouped.unstack()

    cropObservations = unstacked.iloc[0].to_numpy()
    weedObservations = unstacked.iloc[1].to_numpy()

    adfCrop = adfuller(cropObservations, autolag='t-stat')
    dfADF.at[0, factor] = adfCrop[1]
    #print(f"Crop {cropObservations} ADF: {adf}")
    adfWeed = adfuller(weedObservations, autolag='t-stat')
    #print(f"Weed {weedObservations} ADF: {adf}")
    dfADF.at[1, factor] = adfWeed[1]
    dfADF.at[3, factor] = adfCrop[1] < 0.05 and adfWeed[1] < 0.05

# If the number of columns is greater than the threshold, use a graph
if len(dfADF.columns.tolist()) > arguments.threshold:
    plt.style.use('ggplot')
    counts = dfADF.iloc[3].value_counts()
    legend = [f"Unstable {counts[False]}", f"Stable {counts[True]}"]
    plt.pie(counts, labels=legend)
    plt.title(arguments.caption)
    if arguments.output is not None:
        plt.savefig(arguments.output, bbox_inches="tight")
    else:
        plt.show()
else:
    # Clean up the table by dropping the rows and columns not needed
    dfADF.drop(axis=0, inplace=True, index=[0, 1, 2])
    dfADF.drop(axis=1, inplace=True, columns=[constants.NAME_TYPE])
    table = dfADF.to_latex(longtable=True, label=arguments.label, index=False, caption=(arguments.caption, arguments.caption_short))
    if arguments.output is not None:
        with open(arguments.output, "w") as fd:
            fd.write(table)

    print(f"{table}")

# # Plot the means
# unstacked.plot.barh(title=arguments.title)
# ax = plt.gca()
# xAxisNames = ["Weed", "Crop"]
# ax.set_yticklabels(xAxisNames, rotation='vertical')
# plt.ylabel("Type")
# plt.xlabel(arguments.factor)
# # Form the legend from the index names -- the dates of the observations
# dates = list(dfGroupedByDate.index)
# formatted_dates = [date.strftime('%Y-%m-%d') for date in dates]
# print(f"Dates: {formatted_dates}")
# plt.legend(formatted_dates, bbox_to_anchor=(1.04, 0.5), loc='center left')
# #ax.legend(list(dfGroupedByDate.index), loc='upper left')
#
# if arguments.output is not None:
#     plt.savefig(arguments.output, bbox_inches="tight")
# else:
#     plt.show()

sys.exit(0)
