#
# Various operations on results pickle file
#
import logging
import logging.config
import os.path
from collections import Counter
import math
import glob
import sys
from typing import List

from statsmodels.tsa.stattools import adfuller

import argparse
import pickle
import itertools
import more_itertools
import numpy as np
import pandas as pd
import pandas.errors
from pathlib import Path

import constants
from Selection import AllResults
from Selection import IndividualResult
from Selection import Status
from Selection import Criteria

from Factors import Factors
from Factors import FactorTypes
from Factors import FactorSubtypes

from Classifier import ClassificationTechniques
from Classifier import OCCClassificationTechniques
from Classifier import ClassificationType
from Classifier import ImbalanceCorrection
from Classifier import classifierFactory

import matplotlib.pyplot as plt
import matplotlib as mpl

def similarity(l1: [], l2: []) -> float:
    c1 = Counter(l1)
    c2 = Counter(l2)
    terms = set(c1).union(c2)
    dotprod = sum(c1.get(k, 0) * c2.get(k, 0) for k in terms)
    magA = math.sqrt(sum(c1.get(k, 0) ** 2 for k in terms))
    magB = math.sqrt(sum(c2.get(k, 0) ** 2 for k in terms))
    return dotprod / (magA * magB)


def insertBySimilarity(candidate: IndividualResult, similarityThreshold: float) -> pd.DataFrame:
    return


def compare(a: float, b: float):
    """
    Compare two numbers
    :param a:
    :param b:
    :return: -1 = a < b; 0 = a == b; +1 a > b
    """
    return (a > b) - (a < b)


def insertInTopN(candidate: IndividualResult, criterion: Criteria, topN: [], equivalentN: []) -> []:
    """
    Insert the candidate into the top N or the equivalent N
    :param criterion: The criterion to determine topN membership
    :param candidate:
    :param topN:
    :param equivalentN:
    :return:
    """
    replaced = False
    for i in range(len(topN)):
        # print(f"Compare {candidate.accuracy} vs {topN[i].accuracy}")
        candidateScore = 0
        if criterion == Criteria.ACCURACY:
            candidateScore = compare(float(candidate.accuracy), float(topN[i].accuracy))
        elif criterion == Criteria.AUC:
            candidateScore = compare(float(candidate.auc), float(topN[i].auc))

        if candidateScore == 0:
            # print(f"Found equivalent: {candidate.accuracy} for position {i}")
            equivalentN[i].append(candidate)
            break
        if candidateScore == 1:
            # print(f"Found new max {candidate.accuracy} for position {i}")
            topN[i] = candidate
            equivalentN[i] = []
            break
    return topN, equivalentN


def consolidateResults(allResults: List[AllResults], typeOfClassification: ClassificationType) -> pd.DataFrame:
    """
    Form a dataframe for the results, using only the specified type of classification
    :param allResults: All results
    :param typeOfClassification: Type of classification (Binary or OCC)
    :return: Dataframe of the results
    """
    # All results -- will be a list of dictionaries
    results = []
    approachBinary = [t.name for t in ClassificationTechniques]
    approachOCC = [t.name for t in OCCClassificationTechniques]

    assert(len(allResults) > 0)

    # The outer technique is the assessment technique, like AUC
    for technique in allResults:
        # Each result will have the technique
        for individualResult in technique.results:
            if typeOfClassification == ClassificationType.BINARY and individualResult.technique not in approachBinary:
                continue
            if typeOfClassification == ClassificationType.OCC and individualResult.technique not in approachOCC:
                continue
            results.append(individualResult.toDict())

    df = pd.DataFrame(results)

    return df


def findHighestByCriterion(allResults: List[AllResults], criterion: constants.Scoring, classificationType: ClassificationType) -> pd.DataFrame:
    """
    Find the highest result for a given classification type.
    :param allResults: The results read from a file
    :param criterion: The criterion to use (AUC, F1, etc.)
    :param classificationType: The approach type (binary, OCC)
    :return: A dataframe of the highest result for each classification approach
    """
    all = consolidateResults(allResults, classificationType)
    assert(len(all) > 0)
    # Get the series that defines the locations of the highest per group by the criterion specified
    highestPositions = all.groupby(['Technique'])[criterion.name].idxmax()
    # Create a new dataframe with just those entries
    highest = all.iloc[highestPositions.to_list()]

    return highest


def findHighestScores(allResults: List[AllResults], typeOfClassification: ClassificationType, threshold: float) -> pd.DataFrame:
    """
    Find the highest scores among all the results.
    :param classificationType: Type of classification (binary, OCC)
    :param threshold: If this is a non-zero value, record the counts close to the maximum, otherwise record the actual value
    :param allResults: All results
    :return: pandas dataframe with a row for each technique
    """
    highestScores = []
    # Each technique (DECISIONTREE, MLP, etc)
    for technique in allResults:
        highestAUC = -1
        highestMAP = -1
        highestRecall = -1
        highestPrecision = -1
        highestF1 = -1
        highestAccuracy = -1

        # For every result in that technique (every combination of parameters)
        for individualResult in technique.results:
            if typeOfClassification == ClassificationType.BINARY and individualResult.technique not in approachBinary:
                continue
            if typeOfClassification == ClassificationType.OCC and individualResult.technique not in approachOCC:
                continue
            if individualResult.auc > highestAUC:
                highestAUC = individualResult.auc
                print(f"Found highest AUC with: {individualResult.parameters}")
            if individualResult.f1 > highestF1:
                highestF1 = individualResult.f1
                print(f"Found highest F1 with: {individualResult.parameters}")
            if individualResult.precision > highestPrecision:
                highestPrecision = individualResult.precision
                print(f"Found highest precision with: {individualResult.parameters}")
            if individualResult.recall > highestRecall:
                highestRecall = individualResult.recall
                print(f"Found highest recall with: {individualResult.parameters}")
            if individualResult.accuracy > highestAccuracy:
                highestAccuracy = individualResult.accuracy
                print(f"Found highest accuracy with: {individualResult.parameters}")
            if individualResult.map > highestMAP:
                highestMAP = individualResult.map
                print(f"Found highest MAP with: {individualResult.parameters}")

        # Now we have the maximum values for a technique.

        # Walk through the list again to find the count of the values close to that threshold
        if threshold > 0.0:
            countAUC = 0
            countF1 = 0
            countPrecision = 0
            countRecall = 0
            countMAP = 0
            countAccuracy = 0

            for individualResult in technique.results:
                if math.isclose(highestAUC, individualResult.auc, rel_tol=threshold):
                    countAUC += 1
                if math.isclose(highestPrecision, individualResult.precision, rel_tol=threshold):
                    countPrecision += 1
                if math.isclose(highestRecall, individualResult.recall, rel_tol=threshold):
                    countRecall += 1
                if math.isclose(highestF1, individualResult.f1, rel_tol=threshold):
                    countF1 += 1
                if math.isclose(highestMAP, individualResult.map, rel_tol=threshold):
                    countMAP += 1
                if math.isclose(highestAccuracy, individualResult.accuracy, rel_tol=threshold):
                    countAccuracy += 1
                nameOfTechnique = individualResult.technique

            # Each entry is an absolute count. Convert them to %
            percentAUC = (countAUC / len(technique.results)) * 100
            percentF1 = (countF1 / len(technique.results)) * 100
            percentPrecision = (countPrecision / len(technique.results)) * 100
            percentRecall = (countRecall / len(technique.results)) * 100
            percentMAP = (countMAP / len(technique.results)) * 100
            percentAccuracy = (countAccuracy / len(technique.results)) * 100
            #
            if typeOfClassification == ClassificationType.BINARY and individualResult.technique not in approachBinary:
                continue
            if typeOfClassification == ClassificationType.OCC and individualResult.technique not in approachOCC:
                continue
            highestScores.append({'Technique': nameOfTechnique, 'AUC': percentAUC, 'Precision': percentPrecision,
                                  'Recall': percentRecall, 'F1': percentF1})

        # No threshold, record the actual value
        else:
            highestScores.append({'Technique': technique.technique, 'AUC': highestAUC, 'Precision': highestPrecision,
                                  'Recall': highestRecall, 'F1': highestF1})

    # Construct a dataframe from the scores
    df = pd.DataFrame(highestScores)

    return df


def formCaption(rawCaption: str, threshold: float):
    """
    Form the caption for a table, replacing the metacharacters with the threshold
    :param rawCaption: The caption with %T is to be replaced with threshold
    :param threshold:
    :return:
    """
    percentage = f"{threshold:.0%}".replace("%", "\\%")
    caption = rawCaption.replace("%T", str(threshold))
    return caption


parser = argparse.ArgumentParser("Results file operation")

FORMAT_LATEX = "latex"
FORMAT_TEXT = "text"
FORMAT_CSV = "csv"

# ACCURACY, AUC. etc.
criteriaChoices = [e.name for e in Criteria]
criteriaChoices.append(constants.NAME_ALL)

attributeTypes = [e for e in FactorTypes]
attributeChoices = [e.name for e in FactorTypes]
attributeChoices.append(constants.NAME_ALL)

attributeSubtypes = [e for e in FactorSubtypes]
attributeSubtypeChoices = [e.name for e in FactorSubtypes]
attributeSubtypeChoices.append(constants.NAME_ALL)

classifications = [e for e in ClassificationTechniques]
classificationChoices = [e.name for e in ClassificationTechniques]
classificationChoices.append(constants.NAME_ALL)

OCCClassifications = [t for t in OCCClassificationTechniques]
OCCClassificationChoices = [t.name for t in OCCClassificationTechniques]

# The type of the classification, i.e., OCC or Binary
techniqueChoices = [t.name.lower() for t in ClassificationType]
techniqueChoices.append(constants.NAME_ALL)

imbalanceNames = [e.name for e in ImbalanceCorrection]

tableItems = ["parameters", "auc", "f1", "precision", "recall", "map"]
defaultTableItems = ["parameters", "auc"]

parser.add_argument("-c", "--computed", action="store", required=True,
                    help="Pickle format file with computed parameters")
parser.add_argument("-cr", "--criteria", action="store", required=False, default="AUC", choices=criteriaChoices,
                    help="Criteria to determine maximum")
parser.add_argument("-cl", "--long", action="store", required=False, default="Long Caption",
                    help="(latex) Table long caption")
parser.add_argument("-cs", "--short", action="store", required=False, default="Short Caption",
                    help="(latex) Table short caption")
parser.add_argument("-te", "--technique", action="store", required=False, default=ClassificationType.BINARY.name.lower(), choices=techniqueChoices,
                    help="Classification Approach")
parser.add_argument("-df", "--data", action="store", help="Name of the data in CSV format (required for --roc")
parser.add_argument("-f", "--factors", action="store_true", required=False, default=False, help="Show only the factors")
parser.add_argument("-l", "--label", action="store", required=False, default="table:xxxx",
                    help="(latex) Label for generated table")
parser.add_argument("-lg", "--logging", action="store", required=False, default="logging.ini",
                    help="logging configuration")
parser.add_argument("-n", "--n", action="store", required=False, default=-1, type=int, help="Number of combinations")
parser.add_argument("-fo", "--format", action="store", required=False, default=FORMAT_LATEX,
                    choices=[FORMAT_LATEX, FORMAT_TEXT, FORMAT_CSV], help="Output format")
parser.add_argument("-od", "--directory", action="store", required=True, help="Results file directory")
# Items to include in the table -- for example --include auc,f1,precision would include only those as headers
parser.add_argument("-i", "--include", action="store", required=False, nargs='*', choices=tableItems,
                    default=defaultTableItems, help="Include in table")

options = parser.add_mutually_exclusive_group()
# The prefix needs a bit of explanation -- the result file has ALL the five evaluation criteria (auc, f1, map, precision, recall)
# So this should be maricopa_auc, not maricopa, as the auc file will have everything
options.add_argument("-p", "--prefix", action="store", help="File prefix")
options.add_argument("-r", "--results", action="store", help="Pickle file for technique")

# What to produce
parser.add_argument("-su", "--summary", action="store_true", required=False, help="Show a summary of the results")
operations = parser.add_mutually_exclusive_group()
operations.add_argument("-si", "--similarity", action="store", required=False, type=float,
                        help="Show items where similarity matches")
operations.add_argument("-ct", "--close", action="store_true", required=False, default=False,
                        help="Show count of results close to maximum")
operations.add_argument("-roc", "--roc", action="store_true", required=False, default=False, help="Show ROC curves")
operations.add_argument("-rt", "--roc-thresholds", action="store_true", required=False, default=False, help="Produce table of thresholds based on ROC curves")
operations.add_argument("-table", "--table", action="store_true", required=False, default=False, help="Generate results table")
operations.add_argument("-pc", "--parameter-counts", action="store_true", required=False, default=False, help="Generate parameter counts table")

# Produce seasonality comparison table instead of just the attributes
parser.add_argument("-se", "--seasonality", action="store_true", required=False, default=False,
                    help="Show seasonality instead of attribute names")
parser.add_argument("-so", "--source", action="store", required=False, help="The original data for these results")

# The --compare option will produce a table that contains only the before and after scores between the two files
# For instance the first file was uncorrected, and the second was corrected.
parser.add_argument("-co", "--compare", action="store_true", required=False, default=False,
                    help="Compare results to specified set")
parser.add_argument("-t", "--target", action="store", required=False,
                    help="Compare results to this file (required if --compare is")
parser.add_argument("-th", "--threshold", action="store", required=False, default=3e-9, type=float,
                    help="Threshold for --close option")

parser.add_argument("-graph", "--graph", action="store", required=False, help="Output file for graph")
parser.add_argument("-o", "--output", action="store", required=False, help="Output file for table")

# The classification techniques
# Don't make this selectable for now -- not needed
# parser.add_argument("-s", "--subtypes", action="store", required=False, default=constants.NAME_ALL, nargs='*', choices=attributeSubtypeChoices, help="Attribute types used")

# The parameter types and subtypes
parser.add_argument("-s", "--subtypes", action="store", required=False, nargs='*', choices=attributeSubtypeChoices,
                    help="Attribute types used")
parser.add_argument("-ty", "--types", action="store", required=False, nargs='*', choices=attributeChoices,
                    help="Attribute types used")

arguments = parser.parse_args()

# Column names in the output
niceColumnNames = {"_accuracy": "Accuracy",
                   "_auc": "AUC",
                   "_precision": "Precision",
                   "_recall": "Recall",
                   "_f1": "F1",
                   "_parameters": "Parameters"}

if arguments.seasonality:
    if arguments.source is None:
        print("Source name must be specified if seasonality is")
        sys.exit(-1)
    if not os.path.isfile(arguments.source):
        print(f"Unable to access source data: {arguments.source}")
        sys.exit(-1)

if arguments.compare:
    if arguments.target is None:
        print("Target must be specified if compare is.")
        sys.exit(-1)

if arguments.output is not None:
    if os.path.isfile(arguments.output):
        print(f"Output file {arguments.output} exists. Will not overwrite")
        sys.exit(-1)

if arguments.roc and arguments.graph is not None:
    if os.path.isfile(arguments.graph):
        print(f"Output file {arguments.graph} exists. Will not overwrite")
        sys.exit(-1)
if arguments.roc and arguments.graph is None:
    print("Graph output file must be specified for ROC")
    sys.exit(-1)
if arguments.roc and arguments.data is None:
    print("Data file must be specified for ROC")
    sys.exit(-1)
if arguments.roc_thresholds and arguments.output is None:
    print("Table output file must be specified if roc-threshold is")
    sys.exit(-1)

if arguments.data:
    if not os.path.isfile(arguments.data):
        print(f"Unable to access data file: {arguments.data}")
        sys.exit(-1)

# The types and subtypes to be considered. If a subset is not specified, use all of them.
if arguments.types is not None:
    if "all" in arguments.types:
        types = attributeTypes
    else:
        types = [FactorTypes[e] for e in arguments.types]
else:
    types = attributeTypes

if arguments.subtypes is not None:
    if "all" in arguments.subtypes:
        subtypes = attributeSubtypes
    else:
        subtypes = [FactorSubtypes[e] for e in arguments.subtypes]
else:
    subtypes = attributeSubtypes

criteria = Criteria[arguments.criteria]

# Initialize logging
if os.path.isfile(arguments.logging):
    logging.config.fileConfig(arguments.logging)
    log = logging.getLogger("jetson")
else:
    print(f"Unable to access logging configuration: {arguments.logging}")
    exit(-1)

# The similarity to the set of selected parameters
if arguments.similarity is not None and arguments.computed is None:
    print("Computed parameters must be specified if similarity is")
    exit(-1)

# The directory where the results can be found
if arguments.directory is not None:
    if not os.path.isdir(arguments.directory):
        print(f"Unable to access directory: {arguments.directory}")
        exit(-1)
    else:
        baseDir = arguments.directory
else:
    baseDir = "."

# Find data files for each technique
files = []
if arguments.prefix is not None:
    files = glob.glob(os.path.join(baseDir, arguments.prefix + '_*.pickle'))
    if len(files) == 0:
        print(f"Unable to find any result files with prefix {arguments.prefix}")
        exit(-1)
elif arguments.results is not None:
    # Specified a specific results file
    files = [arguments.results]
else:
    print("Must specify either --results or --prefix for result files")

# Load the results from all the <prefix>_*.pickle files, or just the one the user specified
allResults = []
allResultsDF = pd.DataFrame()
for file in files:
    components = Path(file).stem.split('_')
    if len(components) == 5:
        allResultsForTechnique = AllResults(technique=components[2], correction=components[4])
    else:
        allResultsForTechnique = AllResults(technique=components[2])

    print(f"Loading: {file}")
    allResultsForTechnique.load(file)
    resultsDF = allResultsForTechnique.asDataframe()
    # Find the max value, columns are Technique, AUC, F1, Recall, Precision, Parameters
    maximum = resultsDF.loc[resultsDF[constants.Scoring.AUC.name].idxmax()]
    #print(f"AUC: {maximum[1]} F1: {maximum[2]} Recall: {maximum[3]} Precision: {maximum[4]} Parameters: {maximum[5]}")
    allResultsDF = allResultsDF.append(resultsDF)
    allResults.append(allResultsForTechnique)

# Overkill, perhaps, but sometimes it is initial capital (Precision), sometimes all caps (AUC)
readableNames = {
    'technique': {'name': 'Technique', 'format': 'l'},
    'auc': {'name': 'AUC', 'format': 'r'},
    'f1': {'name': 'F1', 'format': 'r'},
    'precision': {'name': 'Precision', 'format': 'r'},
    'recall': {'name': 'Recall', 'format': 'r'},
    'parameters': {'name': 'Parameters', 'format': 'l'}
}

#
# The parameter-counts requires the same information as the summary or detail table does
#
if arguments.table or arguments.parameter_counts:
    highestScores = findHighestByCriterion(allResults, constants.Scoring.AUC, ClassificationType[arguments.technique.upper()])
    columnNames = highestScores.columns.to_list()
    columnsToShow = []
    columnsAliases = []
    columnsInTable = arguments.include
    columnFormat = ""

    # If the user did not have 'Technique' as a column, force it
    if constants.NAME_TECHNIQUE.lower() not in arguments.include:
        columnsInTable.insert(0, constants.NAME_TECHNIQUE)

    # Step through the columns, inserting the name and format required
    for column in columnsInTable:
        try:
            columnsToShow.append(readableNames[column.lower()]['name'])
            columnsAliases.append(readableNames[column.lower()]['name'])
            columnFormat += readableNames[column.lower()]['format']
        except KeyError:
            print(f"Unable to find column: {column}")
            sys.exit(-1)


    longCaption = arguments.long
    shortCaption = arguments.short
    # Convert the technique, as errors are encountered it this is not done
    highestScores['Technique'] = highestScores['Technique'].astype('string')

    # The user wants the table produced
    if arguments.table:

        # The parameters are all series, so convert them to space separated strings
        highestScores['Parameters'] = highestScores['Parameters'].apply(lambda x: ' '.join(x))

        # If the user specified that parameters are to be shown, insert keywords that can be replaced
        if constants.NAME_PARAMETERS.lower() in arguments.include:
            highestScores['Parameters'] = 'MINIPAGESTART ' + highestScores['Parameters'] + ' MINIPAGEEND'

        # If the column width is not set to a high value, the output is truncated
        with pd.option_context("max_colwidth", 1000):
            dfLatex = highestScores.to_latex(longtable=True, header=columnsAliases, index_names=False, index=False,
                                             caption=(longCaption, shortCaption), float_format='%0.2f',
                                             label=arguments.label, columns=columnsToShow, column_format=columnFormat)

        if constants.NAME_PARAMETERS.lower() in arguments.include:
            dfLatex = dfLatex.replace('MINIPAGESTART', '\\begin{minipage}[t]{0.5\\textwidth}')
            dfLatex = dfLatex.replace('MINIPAGEEND \\\\', '\\end{minipage} \\\\ \\midrule')

        # If the user specified a file, write it there, otherwise stdout
        if arguments.output is not None:
            f = open(arguments.output, 'w')
            f.write(dfLatex)
            f.close()
        else:
            print(f"{dfLatex}")

    # Produce a table that has the number of times a specific parameter is used in the classification

    if arguments.parameter_counts:
        # The list of all unique parameters in all the classifications
        uniqueParameters = set()
        for index, row in highestScores.iterrows():
            parameters = list(row['Parameters'])
            [uniqueParameters.add(parameter) for parameter in parameters]
        print(f"Unique set: {uniqueParameters}")
        # Convert to list and dataframe, assigning an initial count
        countsDict = dict.fromkeys(list(uniqueParameters), 0)
        counts = pd.DataFrame.from_dict(countsDict, orient='index', columns=['Count'])
        columnsAliases = ['Parameter', f'Of {len(highestScores)} models']

        # Iterate over the parameters, counting usage
        for index, row in highestScores.iterrows():
            for parameter in row['Parameters']:
                counts.loc[parameter, 'Count'] += 1

        counts.reset_index(inplace=True)
        dfLatex = counts.to_latex(longtable=True, header=columnsAliases, index_names=False, index=False,
                                  caption=(longCaption, shortCaption), float_format='%0.2f',
                                  label=arguments.label)
        if arguments.output is not None:
            f = open(arguments.output, "w")
            f.write(dfLatex)
            f.close()
        else:
            print(f"{dfLatex}")


    sys.exit(0)

# The parameters selection is really only required in producing the similarity report or to just print them out,
# TODO: make this required only if similarity is
try:
    computed = pd.read_pickle(arguments.computed)

    # Just show the calculated parameters and exit
    if arguments.factors:
        if computed is not None:
            print("----- F A C T O R S -----")
            print(f"{computed}")
            print("----- F A C T O R S -----")
            exit(0)
except FileNotFoundError:
    print(f"Unable to access parameter file: {arguments.computed}")
    exit(-1)

# The names of the selection techniques
parameters = computed.index
scoreNames = ["Precision", "Recall", "F1", "MAP", "AUC"]


# Some of the dataframes have multiple rows -- set these to the averages
def averageScores(finalSubset: pd.DataFrame) -> pd.DataFrame:
    """
    Return a new dataframe with the averages of the dataframe supplied
    :param finalSubset: a dataframe with multiple rows
    :return: a new dataframe with a single row
    """
    # This is a bit of a hack -- just treat the technique and parameter as string, and everything else is a float
    # This is to address the problem where constant values are type Object, not float
    # stringColumns = ['Technique', 'Parameters', 'Correction']
    stringColumns = ['Technique', 'Parameters']
    typeConversions = {}
    for column in finalSubset.columns:
        if column not in stringColumns:
            typeConversions[column] = float
        else:
            typeConversions[column] = object
    finalSubset = finalSubset.astype(typeConversions)

    # Copy the frame, but drop the rows
    averaged = finalSubset.copy()
    averaged.drop(index=finalSubset.index, axis=0, inplace=True)

    # Calculate means of the numerics
    averages = finalSubset.mean(numeric_only=True)

    # The new frame has the first column set to the technique name, and the remainder to the averages
    # Original
    # averaged.loc[0] = [finalSubset.at[0, 'Technique']] + list(averages)
    # Catch the case where the averages are negative
    if len(finalSubset.at[0, 'Parameters']) == 0:
        finalSubset.at[0, 'Parameters'] = 'UNKNOWN'

    if averages[0] > 0 and len(averages) == 5:
        averaged.loc[0] = [finalSubset.at[0, 'Technique']] + list(averages) + [finalSubset.at[0, 'Parameters']]

    # There is a bug in the code -- if the column is constant, it is of type object, not float
    if len(averages) != 5:
        print(f"Technique {finalSubset.at[0, 'Technique']} has {len(averages)}")

    return averaged


allFactors = Factors()


def replaceParametersWithSeasonality(results, sourceDF):
    # adfCrop = adfuller(cropObservations, autolag='t-stat')
    resultsDF = results.copy()
    # Separate into weed and crop segments
    crop = sourceDF[sourceDF['type'] == 0]
    weeds = sourceDF[sourceDF['type'] == 1]

    seasonality = []

    # Calculate the ADF for one column with adfTest = adfuller(sourceDF['shape_index']
    for idx in range(len(resultsDF)):
        parameters = resultsDF.iloc[idx]['Parameters'].split()
        isSeasonal = []
        for parameter in parameters:
            # Not quite correct -- treats everything as crop
            adfCrop = adfuller(crop[parameter])
            adfWeed = adfuller(weeds[parameter])
            isSeasonal.append(adfCrop[1] < 0.05 and adfWeed[1] < 0.05)

        # resultsDF.iloc[idx]['Parameters'] = isSeasonal
        stringList = [str(b) for b in isSeasonal]
        seasonalString = " ".join(stringList)
        seasonality.append(seasonalString)

    resultsDF.drop(axis=1, inplace=True, columns=["Parameters"])
    resultsDF["Parameter Stability"] = seasonality
    return resultsDF


if arguments.summary:
    columnNames = ['Technique']
    columnNames.extend(list(scoreNames))
    # columnNames.extend(["Correction"])
    columnNames.extend(["Parameters"])

    resultsDF = pd.DataFrame(columns=columnNames)
    approachBinary = [t.name for t in ClassificationTechniques]
    approachOCC = [t.name for t in OCCClassificationTechniques]

    for technique in allResults:

        # print(f"Technique {technique.technique} Correction {technique.correction}")
        resultsSubset = pd.DataFrame(columns=columnNames)
        highestAUC = -1
        highestMAP = -1
        highestF1 = -1
        highestPrecision = -1
        highestRecall = -1

        # Results whose factors match the specification
        matchingRows = 0

        # Create 5 entries for a given technique
        for scores in scoreNames:
            # resultsSubset.loc[len(resultsSubset)] = [technique.technique, -1, -1, -1, -1, -1, technique.correction, technique.parameters]
            resultsSubset.loc[len(resultsSubset)] = [technique.technique, -1, -1, -1, -1, -1, technique.parameters]

        # Iterate over the results, replacing as needed
        for result in technique.results:
            print(f"Technique {result.technique} Binary: {approachBinary}")
            if ClassificationType[arguments.technique.upper()] == ClassificationType.BINARY and result.technique not in approachBinary:
                continue
            if ClassificationType[ arguments.technique.upper()] == ClassificationType.OCC and result.technique not in approachOCC:
                continue
            print(f"Result: {result}")
            # Catch the case where the computation is not complete.
            if result.status != Status.COMPLETED:
                print(f"Entry is not complete: {result}")
                continue
            # Determine if the parameter set is composed entirely of the subset specified
            if not allFactors.composedOfTypes(types, result.parameters):
                continue
            if not allFactors.composedOfSubtypes(subtypes, result.parameters):
                continue
            elif result.status != Status.COMPLETED:
                continue
            else:
                matchingRows += 1

            # print(f"{result.auc},{result.accuracy}")
            allParameters = ' '.join(result.parameters)
            # row = [result.technique, result.precision, result.recall, result.f1, result.map, result.auc, result.correction, allParameters]
            row = [result.technique, result.precision, result.recall, result.f1, result.map, result.auc, allParameters]
            replacementIndex = -1

            # The highest scores seen for each correction technique
            # highestAUC = {constants.NAME_NONE: -1, ImbalanceCorrection.SMOTE.name: -1, ImbalanceCorrection.ADASYN.name: -1, ImbalanceCorrection.BORDERLINE.name: -1, ImbalanceCorrection.KMEANS.name: -1, ImbalanceCorrection.SVM.name: -1, ImbalanceCorrection.SMOTETOMEK.name: -1, ImbalanceCorrection.SMOTEENN.name: -1}
            # highestMAP = highestAUC.copy()
            # highestF1 = highestAUC.copy()
            # highestPrecision = highestAUC.copy()
            # highestRecall =  highestAUC.copy()
            closestAUC = 0
            closestMAP = 0
            closestF1 = 0
            closestPrecision = 0
            closestRecall = 0

            highestAUC = -1
            highestMAP = -1
            highestF1 = -1
            highestPrecision = -1
            highestRecall = -1

            # Iterate over the results to find the highest score for the results
            for index, techniqueItem in resultsSubset.iterrows():
                # print(f"{techniqueItem['Technique']} + {techniqueItem['Correction']}: AUC: {techniqueItem['AUC']} Parameters: {techniqueItem['Parameters']}")
                print(
                    f"{techniqueItem['Technique']} : AUC: {techniqueItem['AUC']} Parameters: {techniqueItem['Parameters']}")
                # if result.auc > techniqueItem["AUC"] and result.auc > highestAUC[technique["Correction"]]:
                if arguments.criteria == "AUC":
                    if result.auc > techniqueItem["AUC"] and result.auc > highestAUC:
                        print(f"AUC for index {index} {result.auc} vs {techniqueItem['AUC']}")
                        replacementIndex = index
                        highestAUC = result.auc
                        continue
                # #if result.precision > techniqueItem["Precision"] and result.precision > highestPrecision[technique["Correction"]]:
                # if result.precision > techniqueItem["Precision"] and result.precision > highestPrecision:
                #     #print(f"Precision for index {index} {result.precision} vs {techniqueItem['Precision']}")
                #     replacementIndex = index
                #     highestPrecision = result.precision
                #     continue
                # #if result.recall > techniqueItem["Recall"] and result.recall > highestRecall[technique["Correction"]]:
                # if result.recall > techniqueItem["Recall"] and result.recall > highestRecall:
                #     #print(f"Recall for index {index} {result.recall} vs {techniqueItem['Recall']}")
                #     replacementIndex = index
                #     highestRecall = result.recall
                #     continue
                # #if result.f1 > techniqueItem["F1"] and result.f1 > highestF1[technique["Correction"]]:
                # if result.f1 > techniqueItem["F1"] and result.f1 > highestF1:
                #     #print(f"F1 for index {index} {result.f1} vs {techniqueItem['F1']}")
                #     replacementIndex = index
                #     highestPrecision = result.f1
                #     continue
                # #if result.map > techniqueItem["MAP"] and result.map > highestMAP[technique["Correction"]]:
                # if result.map > techniqueItem["MAP"] and result.map > highestMAP:
                #     #print(f"MAP for index {index} {result.map} vs {techniqueItem['MAP']}")
                #     replacementIndex = index
                #     highestMAP = result.map
                #     continue
                # replaceCurrentRow = highestPrecision == result.precision and highestRecall == result.recall and highestF1 == result.f1 and highestMAP == result.map
            if replacementIndex != -1:
                print(f"Replacement index {replacementIndex}")
                resultsSubset.loc[replacementIndex] = row
                replacementIndex = -1
        # Now we have entries where some rows are not the top in anything or are tied for the top
        finalSubset = pd.DataFrame(columns=columnNames)
        for score in scoreNames:
            resultsSubset = resultsSubset.convert_dtypes()
            idx = resultsSubset[score].idxmax()
            row = resultsSubset.iloc[idx]
            finalSubset.loc[len(finalSubset)] = row

        if matchingRows > 0:
            finalSubset.drop_duplicates(inplace=True)
            # If the length of the final subset is > 1, then take the average of what is found.
            if len(finalSubset) > 1:
                finalSubset = averageScores(finalSubset)
            print(f"Appending {len(finalSubset)} to dataframe")
            resultsDF = resultsDF.append(finalSubset)

    # For results that are close to the maximum, we have to go through the list again, now that the maximum is found
    if arguments.close:
        # Find the counts of the highest scores that are within the threshold of the maximum
        highestScores = findHighestScores(allResults, ClassificationType[arguments.technique.upper()], threshold=arguments.threshold)
        highestScores.sort_values(by=["AUC"], inplace=True)

        # Form the caption, replacing text for the threshold
        longCaption = formCaption(arguments.long, arguments.threshold)
        shortCaption = formCaption(arguments.short, arguments.threshold)

        # We need only two columns, the technique and % close to the maximum
        columnFormat = 'lrrrr'
        columnsToShow = ['Technique', 'AUC', 'Precision', 'Recall', 'F1']
        columnsAliases = ['Technique', '% AUC', '% Precision', '% Recall', '% F1']
        dfLatex = highestScores.to_latex(longtable=True, header=columnsAliases, index_names=False, index=False,
                                         caption=(longCaption, shortCaption), float_format='%0.2f',
                                         label=arguments.label, columns=columnsToShow, column_format=columnFormat)
        if arguments.output is not None:
            f = open(arguments.output, "w")
            f.write(f"{dfLatex}")
            f.close()
        print(f"{dfLatex}")
        sys.exit(0)
        #
        # closeResults = [
        #     {'Technique': ClassificationTechniques.SVM.name, 'AUC': 0, 'Precision': 0, 'Recall': 0, 'MAP': 0},
        #     {'Technique': ClassificationTechniques.MLP.name, 'AUC': 0, 'Precision': 0, 'Recall': 0, 'MAP': 0},
        #     {'Technique': ClassificationTechniques.LDA.name, 'AUC': 0, 'Precision': 0, 'Recall': 0, 'MAP': 0},
        #     {'Technique': ClassificationTechniques.RANDOMFOREST.name, 'AUC': 0, 'Precision': 0, 'Recall': 0, 'MAP': 0},
        #     {'Technique': ClassificationTechniques.EXTRA.name, 'AUC': 0, 'Precision': 0, 'Recall': 0, 'MAP': 0},
        #     {'Technique': ClassificationTechniques.GRADIENT.name, 'AUC': 0, 'Precision': 0, 'Recall': 0, 'MAP': 0},
        #     {'Technique': ClassificationTechniques.DECISIONTREE.name, 'AUC': 0, 'Precision': 0, 'Recall': 0, 'MAP': 0},
        #     {'Technique': ClassificationTechniques.KNN.name, 'AUC': 0, 'Precision': 0, 'Recall': 0, 'MAP': 0},
        #     {'Technique': ClassificationTechniques.LOGISTIC.name, 'AUC': 0, 'Precision': 0, 'Recall': 0, 'MAP': 0}
        # ]
        # counts = pd.DataFrame(columns=['Technique', 'AUC', 'Precision', 'Recall', 'MAP'])
        # counts = pd.DataFrame.from_dict(closeResults)
        # counts.set_index('Technique', inplace=True)
        #
        # # For each of the ML techniques (DECISIONTREE, EXTRA, etc.)
        # for technique in allResults:
        #     # For each results
        #     numberOfResults = 0
        #     techniqueName = ""
        #     for result in technique.results:
        #         techniqueName = result.technique
        #         techniqueMax = resultsDF.loc[resultsDF['Technique'] == result.technique]
        #         if math.isclose(techniqueMax.iloc[0][arguments.criteria], result.auc, rel_tol=9e-3):
        #             #print(f"{result.auc} is close to {techniqueMax.iloc[0][arguments.criteria]}")
        #             #print(f"{result.technique} has multiple items close to max")
        #             closeResults[result.technique] += 1
        #     # The closeResults dictionary holds the raw counts, so convert them to percentages
        #     closeResults[techniqueName] = (closeResults[techniqueName] / len(technique.results)) * 100
        # closeDF = pd.DataFrame.from_dict(closeResults, orient='index', columns=["Percentage"])
        # closeDF.sort_values(by=["Percentage"], inplace=True)
        # closeDF = closeDF.reset_index()
        #
        # longCaption = arguments.long
        # shortCaption = arguments.short
        # dfLatex = closeDF.to_latex(longtable=True, index_names=False, index=False, caption=(longCaption, shortCaption), float_format='%.3f', label=arguments.label, column_format='ll')
        # dfLatex = dfLatex.replace("index", "Technique")
        # if arguments.output is not None:
        #     f = open(arguments.output, "w")
        #     f.write(f"{dfLatex}")
        #     f.close()
        # print(f"{dfLatex}")

    #
    # R O C
    #
    # Show the ROC curves for the supported algorithms.
    # As the results files do not contain the FPR and NPR data we need, this will re-do the scoring.

    if arguments.roc or arguments.roc_thresholds:
        rocResults = {}
        highestScores = findHighestByCriterion(allResults, constants.Scoring.AUC, ClassificationType[arguments.technique.upper()])
        # This dataframe will have as many entries as there are techniques.

        # Use this frame for the thresholds
        thresholds = pd.DataFrame(columns=['Technique', 'Threshold'])
        thresholds['Technique'] = [t.name for t in ClassificationTechniques]
        thresholds.set_index(['Technique'], inplace=True)

        # Do the classifications again so we can get the FPR and TPR data
        for index, result in highestScores.iterrows():
            print(f"Classify with {result['Technique']} using {result['Parameters']}")
            classifier = classifierFactory(result['Technique'])
            classifier.selections = result['Parameters']
            classifier.correct = False
            classifier.load(arguments.data, stratify=False)
            classifier.createModel(True)
            classifier.assess()
            classifier.name = result['Technique']

            # Modified from tutorial from Digital Sreeni for finding the optimal threshold
            # https://www.youtube.com/watch?v=Joh3LOaG8Q0

            i = np.arange(len(classifier.tpr))
            roc = pd.DataFrame({'tf': pd.Series(classifier.tpr - (1 - classifier.fpr), index=i),
                                'thresholds': pd.Series(classifier.thresholds, index=i)})
            ideal_roc_thresh = roc.iloc[
                (roc.tf - 0).abs().argsort()[:1]]  # Locate the point where the value is close to 0
            loc = ideal_roc_thresh['thresholds'].index[0]
            print( f"Ideal threshold: {ideal_roc_thresh.iloc[0]['thresholds']}  at index {ideal_roc_thresh['thresholds'].index[0]}")
            ideal = (classifier.fpr[loc], classifier.tpr[loc])

            rocResults[classifier.name] = {'FPR': classifier.fpr, "TPR": classifier.tpr, "AUC": classifier.auc, "threshold": ideal}
            print(f"Technique {classifier.name}: Threshold: {ideal}")

            thresholds.at[classifier.name, 'Threshold'] = ideal_roc_thresh.iloc[0]['thresholds']

        if arguments.roc_thresholds:
            # Convert the thresholds to latex if requested
            longCaption = arguments.long
            shortCaption = arguments.short
            # If we don't explicitly convert the Threshold column to a float64, we see problems
            thresholds['Threshold'] = thresholds['Threshold'].astype('float64')
            # If we don't reset the index to numeric and then don't show it, the headers are messed up.
            thresholds.reset_index(inplace=True)

            f = open(arguments.output, "w")
            f.write(f"{thresholds.to_latex(longtable=True, index=False, caption=(longCaption, shortCaption), float_format='%.3f', label=arguments.label)}")
            f.close()

        if arguments.roc:
            # Visualize the results
            plt.style.use('ggplot')
            plt.rc('font', family='Times New Roman')
            colors = {'KNN': 'black', 'EXTRA': 'red', 'GRADIENT': 'sienna', 'LDA': 'orange', 'LOGISTIC': 'yellowgreen',
                      'MLP': 'blue', 'RANDOMFOREST': 'darkviolet', 'SVM': 'slategrey', 'OCC': 'lightpink', 'DECISIONTREE': 'turquoise'}
            for techniqueName, rates in rocResults.items():
                techniqueAndAUC = f"{rates['AUC']:.2f} {techniqueName}"
                try:
                    plt.plot(rates["FPR"], rates["TPR"], label=techniqueAndAUC, color=colors[techniqueName])
                except KeyError as err:
                    print(f"Unable to find color for: {techniqueName}")
            plt.legend(loc='lower right')
            # The chance line
            plt.plot([0, 1], [0, 1], 'r--')
            # Setting the limit to a value slightly outside the max makes the plot look better
            plt.xlim([0, 1.03])
            plt.ylim([0, 1.03])
            plt.ylabel('True Positive Rate')
            plt.xlabel('False Positive Rate')
            plt.title(f'{arguments.long}')


            # If the user specified and output file, save the plot, otherwise just show it
            if arguments.graph is None:
                plt.show()
            else:
                plt.savefig(arguments.graph)
                #plt.show()



        sys.exit(0)

    if matchingRows > 0 and not arguments.close:
        print(f"-- begin latex for summary ---")
        longCaption = arguments.long
        shortCaption = arguments.short
        # resultsDF = resultsDF.convert_dtypes()

        # Frame cleanup and explicitly set type
        # If we don't explicitly convert the Technique column to a string, we see problems
        resultsDF['Technique'] = resultsDF['Technique'].astype('string')
        # All other columns except for Technique and Parameters should be float64, or there are problems
        stringColumns = ['Technique', 'Parameters']
        for column in resultsDF.columns:
            if column not in stringColumns:
                resultsDF[column] = resultsDF[column].astype('float64')

        # resultsDF['Correction'] = resultsDF['Correction'].astype('string')
        # resultsDF.drop(["Technique"], inplace=True, axis=1)
        resultsDF.drop_duplicates(inplace=True)

        # Drop the columns except the technique and parameters
        # whitelist = ['Technique', 'Parameters', 'Correction']
        whitelist = [item.upper() for item in arguments.include]
        whitelist.append("TECHNIQUE")
        print(f"All columns: {resultsDF.columns} Target: {whitelist}")
        # Drop the columns that are not wanted
        for column in resultsDF.columns:
            if column.upper() not in whitelist and column.upper() != arguments.criteria:
                resultsDF.drop(column, inplace=True, axis=1)



        if arguments.seasonality:
            sourceDF = pd.read_csv(arguments.source)
            resultsDF = replaceParametersWithSeasonality(resultsDF, sourceDF)
        else:
            assert 0
            # This code should not be used
            # This is a hack to get the parameters to display correctly.  Put them in a minipage, so first insert strings to replace
            if "parameters".upper() in whitelist:
                # resultsDF['Parameters'] = '\\begin{minipage}[t]{0.5\\textwidth}' + resultsDF['Parameters'] + ' \\end{minipage}'
                resultsDF['Parameters'] = 'MINIPAGESTART ' + resultsDF['Parameters'] + ' MINIPAGEEND'

        # resultsDF['Parameters'] = resultsDF['Parameters'].astype('string')
        # Original
        # print(f"{resultsDF.to_latex(longtable=True, index_names=False, index=False, caption=(longCaption, shortCaption), float_format='%.2f', label=arguments.label)}")
        # The max_colwidth option is needed to avoid the row being truncated
        f = None
        if arguments.output is not None:
            f = open(arguments.output, "w")
        with pd.option_context("max_colwidth", 1000):
            # Formatting table columns based on numbers -- we need technique name and AUC + whatever is specified to include
            formatOfColumns = 'lc'
            for columnPosition in range(len(resultsDF.columns) - 2):
                formatOfColumns += 'l'
            #print(f"Format: {formatOfColumns}")
            if arguments.output is not None:
                latex = resultsDF.to_latex(longtable=True, index_names=False, index=False, caption=(longCaption, shortCaption), float_format='%.2f', label=arguments.label)
                latex = latex.replace('MINIPAGESTART', '\\begin{minipage}[t]{0.5\\textwidth}')
                latex = latex.replace('MINIPAGEEND \\\\', '\\end{minipage} \\\\ \\midrule')
                f.write(f"{latex}")
            # print(f"{resultsDF.to_latex(longtable=True, index_names=False, index=False, caption=(longCaption, shortCaption),  float_format='%.3f', label=arguments.label, column_format=formatOfColumns)}")
        if f is not None:
            f.close()
        print(f"-----------------")
        exit(0)
    else:
        print(f"No result found composed entirely of: {arguments.types}")
        exit(-1)

if arguments.similarity is not None:
    columnNames = ['Technique']  # , 'Accuracy']
    columnNames.append(arguments.criteria)
    columnNames.extend(list(parameters))
    # The resultsSubset frame holds the results and similarity to the parameter selection techniques
    resultsSubset = pd.DataFrame(columns=columnNames)
    resultRows = []
    for technique in allResults:
        for result in technique.results:
            if result.status == Status.COMPLETED:
                row = [result.technique, result.accuracy]
                matched = False
                for parameter in parameters:
                    similarityToSelectionTechnique = similarity(result.parameters, computed.loc[parameter])
                    log.debug( f"Technique: {result.technique} AUC: {result.auc} Similarity: {similarityToSelectionTechnique} {result.parameters} vs {list(computed.loc[parameter])}")
                    if similarityToSelectionTechnique >= arguments.similarity:
                        matched = True
                    # See if we already have the technique in the result set
                    row.append(similarity(result.parameters, computed.loc[parameter]))

                if matched:
                    resultRows.append(row)
                    resultsSubset.loc[len(resultsSubset)] = row
            else:
                log.error(f"Technique is not complete: {result.technique}")

    # Too complicated here -- we don't need to sort through all techniques, as they all have the scores needed.
    # If we don't do this, the results are replicated
    resultsSubset.drop_duplicates(inplace=True)

    print(f"-- begin latex for similarity: {arguments.similarity} --")
    longCaption = arguments.long
    shortCaption = arguments.short
    print(
        f"{resultsSubset.to_latex(longtable=True, index_names=False, index=False, caption=(longCaption, shortCaption), label='table:matches-selection', header=columnNames, float_format='%.2f')}")
    print("-- end latex --")
    exit(0)

if arguments.format is not None:
    if arguments.n == -1:
        for result in allResults:
            result.print()
    else:
        # The top Individual results
        topN = [IndividualResult()] * arguments.n
        # For every one of the topN, allow 100 equivalents
        equivalentN = [list()] * arguments.n
        # equivalentN: list[IndividualResult("XXX")] = list()

        # The lowest and highest results seen
        lowestResultInTopN = 0.0
        highestResultInTopN = 0.0
        lowestResultPosition = 0
        highestPositionUsed = 0
        position = 0
        lowestAccuracy = 100.0
        lowestAccuracyPosition = 0

        # results = allResultsForTechnique.results
        for technique in allResults:
            print(f"Technique: {technique.technique}")
            resultNumber = 0
            for result in technique.results:
                resultNumber += 1
                # print(result)
                if result.status == Status.COMPLETED:
                    topN, equivalentN = insertInTopN(result, criteria, topN, equivalentN)
                else:
                    print(
                        f"Something is wrong: results for #{resultNumber} are not complete. Marked as {result} using {technique.technique}")
        dfTopN = pd.DataFrame(topN)
        # print(f"Top {arguments.n} of {len(results)}")
        if arguments.format == FORMAT_CSV:
            # Column headers
            print("sequence,technique,auc,accuracy,base_similarity,parameters", end="")
            for selection in parameters:
                print(f",{'similarity_' + selection}", end="")
            print()
            # Data for topN and each equivalent
            for i in range(len(topN)):
                if topN[i].status != Status.UNCLAIMED:
                    # print(f"{topN[i]} Equivalents: {len(equivalentN[i])}")
                    print(f"\n{i},{topN[i].technique},{topN[i].auc},{topN[i].accuracy},1,{topN[i].parameters}", end='')
                    for selection in parameters:
                        print(f",{similarity(topN[i].parameters, computed.loc[selection])}", end='')
                    for equivalent in equivalentN[i]:
                        print(
                            f"\n{i},{equivalent.technique},{equivalent.auc},{equivalent.accuracy},{topN[i].similarity(equivalent)},{equivalent.parameters}",
                            end='')
                        for selection in parameters:
                            print(f",{similarity(topN[i].parameters, computed.loc[selection])}", end='')
        elif arguments.format == FORMAT_LATEX:
            # Convert the topN list to a list of dictionaries
            myData = [vars(topN[i]) for i in range(len(topN))]
            dfTopN = pd.DataFrame.from_records(myData)
            columnsToDrop = []
            for column in dfTopN.columns.tolist():
                if column not in niceColumnNames:
                    columnsToDrop.append(column)
            dfTopN.drop(columns=columnsToDrop, inplace=True)
            dfTopN.rename(columns=niceColumnNames, inplace=True)
            print('----- begin latex -----')
            longCaption = "Long Caption"
            shortCaption = "Short Caption"
            print(
                f"{dfTopN.to_latex(longtable=True, index_names=False, index=False, caption=(longCaption, shortCaption), label=arguments.label)}")
            print('----- end latex -------')
        elif arguments.format == FORMAT_TEXT:
            print("Text output not supported")
            exit(-1)
        else:
            print(f"Unknown output format: {arguments.format}")
            # print()

    sys.exit(0)
