#
# Various operations on results pickle file
#
import logging
import logging.config
import os.path
from collections import Counter
import math
import glob

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
        #print(f"Compare {candidate.accuracy} vs {topN[i].accuracy}")
        candidateScore = 0
        if criterion == Criteria.ACCURACY:
            candidateScore = compare(float(candidate.accuracy), float(topN[i].accuracy))
        elif criterion == Criteria.AUC:
            candidateScore = compare(float(candidate.auc), float(topN[i].auc))

        if candidateScore == 0:
            #print(f"Found equivalent: {candidate.accuracy} for position {i}")
            equivalentN[i].append(candidate)
            break
        if candidateScore == 1:
            #print(f"Found new max {candidate.accuracy} for position {i}")
            topN[i] = candidate
            equivalentN[i] = []
            break
    return topN, equivalentN

parser = argparse.ArgumentParser("Results file operation")

FORMAT_LATEX = "latex"
FORMAT_TEXT = "text"
FORMAT_CSV = "csv"

# ACCURACY, AUC. etc.
criteriaChoices = [e.name for e in Criteria]
criteriaChoices.append(constants.NAME_ALL)

attributeTypes = [e for e in FactorTypes]
attributeChoices = [e.name for e in FactorTypes]
#attributeChoices.append(constants.NAME_ALL)

attributeSubtypes = [e for e in FactorSubtypes]
attributeSubtypeChoices = [e.name for e in FactorSubtypes]
attributeSubtypeChoices.append(constants.NAME_ALL)

classifications = [e for e in ClassificationTechniques]
classificationChoices = [e.name for e in ClassificationTechniques]
classificationChoices.append(constants.NAME_ALL)

parser.add_argument("-c", "--computed", action="store", required=True, help="Pickle format file with computed parameters")
parser.add_argument("-cr", "--criteria", action="store", required=False, default="AUC", choices=criteriaChoices, help="Criteria to determine maximum")
parser.add_argument("-cl", "--long", action="store", required=False, default="Long Caption", help="(latex) Table long caption")
parser.add_argument("-cs", "--short", action="store", required=False, default="Short Caption", help="(latex) Table short caption")
parser.add_argument("-f", "--factors", action="store_true", required=False, default=False, help="Show only the factors")
parser.add_argument("-l", "--label", action="store", required=False, default="table:xxxx", help="(latex) Label for generated table")
parser.add_argument("-lg", "--logging", action="store", required=False, default="logging.ini", help="logging configuration")
parser.add_argument("-n", "--n", action="store", required=False, default=-1, type=int, help="Number of combinations")
parser.add_argument("-o", "--output", action="store", required=True, choices=[FORMAT_LATEX, FORMAT_TEXT, FORMAT_CSV], help="Output format")
parser.add_argument("-od", "--directory", action="store", required=True,  help="Results file directory")

options = parser.add_mutually_exclusive_group()
# The prefix needs a bit of explanation -- the result file has ALL the five evaluation criteria (auc, f1, map, precision, recall)
# So this should be maricopa_auc, not maricopa, as the auc file will have everything
options.add_argument("-p", "--prefix", action="store", help="File prefix")
options.add_argument("-r", "--results", action="store", help="Pickle file for technique")

parser.add_argument("-si", "--similarity", action="store", required=False, type=float, help="Show items where similarity matches")
parser.add_argument("-su", "--summary", action="store_true", required=False, help="Show a summary of the results")

# The classification techniques
# Don't make this selectable for now -- not needed
#parser.add_argument("-s", "--subtypes", action="store", required=False, default=constants.NAME_ALL, nargs='*', choices=attributeSubtypeChoices, help="Attribute types used")

# The parameter types and subtypes
parser.add_argument("-s", "--subtypes", action="store", required=False, nargs='*', choices=attributeSubtypeChoices, help="Attribute types used")
parser.add_argument("-ty", "--types", action="store", required=False, nargs='*', choices=attributeChoices, help="Attribute types used")

arguments = parser.parse_args()

# Column names in the output
niceColumnNames = {"_accuracy": "Accuracy",
                   "_auc": "AUC",
                   "_precision": "Precision",
                   "_recall": "Recall",
                   "_f1": "F1",
                   "_parameters": "Parameters"}

# The types and subtypes to be considered. If a subset is not specified, use all of them.
if arguments.types is not None:
    types = [FactorTypes[e] for e in arguments.types]
else:
    types = attributeTypes

if arguments.subtypes is not None:
    subtypes = arguments.subtypes
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
for file in files:
    components = Path(file).stem.split('_')
    allResultsForTechnique = AllResults(technique=components[2])
    allResultsForTechnique.load(file)
    allResults.append(allResultsForTechnique)

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
    # Copy the frame, but drop the rows
    averaged = finalSubset.copy()
    averaged.drop(index=finalSubset.index, axis=0, inplace=True)

    # Calculate means of the numerics
    averages = finalSubset.mean(numeric_only=True)

    # The new frame has the first column set to the technique name, and the remainder to the averages
    # Original
    #averaged.loc[0] = [finalSubset.at[0, 'Technique']] + list(averages)
    # Catch the case where the averages are negative
    if len(finalSubset.at[0, 'Parameters']) == 0:
        finalSubset.at[0, 'Parameters'] = 'UNKNOWN'

    if averages[0] > 0:
        averaged.loc[0] = [finalSubset.at[0, 'Technique']] + list(averages) + [finalSubset.at[0, 'Parameters']]


    return averaged

allFactors = Factors()

if arguments.summary:
    columnNames = ['Technique']
    columnNames.extend(list(scoreNames))
    columnNames.extend(["Parameters"])

    resultsDF = pd.DataFrame(columns=columnNames)

    for technique in allResults:
        print(f"Technique {technique.technique}")
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
            resultsSubset.loc[len(resultsSubset)] = [technique.technique, -1, -1, -1, -1, -1, technique.parameters]

        # Iterate over the results, replacing as needed
        for result in technique.results:
            #print(f"Status: {result.status}")
            # Catch the case where the computation is not complete.
            if result.status != Status.COMPLETED:
                print(f"Entry is not complete: {result}")
                continue
            # Determine if the parameter set is composed entirely of the subset specified
            if not allFactors.composedOfTypes(types, result.parameters):
                continue
            elif result.status != Status.COMPLETED:
                continue
            else:
                matchingRows += 1

            #print(f"{result.auc},{result.accuracy}")
            allParameters = ' '.join(result.parameters)
            row = [result.technique, result.precision, result.recall, result.f1, result.map, result.auc, allParameters]
            replacementIndex = -1
            highestAUC = -1
            highestMAP = -1
            highestF1 = -1
            highestPrecision = -1
            highestRecall = -1

            # Iterate over the results to find the highest score for the results
            for index, techniqueItem in resultsSubset.iterrows():
                if result.auc > techniqueItem["AUC"] and result.auc > highestAUC:
                    #print(f"AUC for index {index} {result.auc} vs {techniqueItem['AUC']}")
                    replacementIndex = index
                    highestAUC = result.auc
                    continue
                if result.precision > techniqueItem["Precision"] and result.precision > highestPrecision:
                    #print(f"Precision for index {index} {result.precision} vs {techniqueItem['Precision']}")
                    replacementIndex = index
                    highestPrecision = result.precision
                    continue
                if result.recall > techniqueItem["Recall"] and result.recall > highestRecall:
                    #print(f"Recall for index {index} {result.recall} vs {techniqueItem['Recall']}")
                    replacementIndex = index
                    highestRecall = result.recall
                    continue
                if result.f1 > techniqueItem["F1"] and result.f1 > highestF1:
                    #print(f"F1 for index {index} {result.f1} vs {techniqueItem['F1']}")
                    replacementIndex = index
                    highestPrecision = result.f1
                    continue
                if result.map > techniqueItem["MAP"] and result.map > highestMAP:
                    #print(f"MAP for index {index} {result.map} vs {techniqueItem['MAP']}")
                    replacementIndex = index
                    highestMAP = result.map
                    continue
                #replaceCurrentRow = highestPrecision == result.precision and highestRecall == result.recall and highestF1 == result.f1 and highestMAP == result.map
            if replacementIndex != -1:
                #print(f"Replacement index {replacementIndex}")
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


    if matchingRows > 0:
        print(f"-- begin latex for summary ---")
        longCaption = arguments.long
        shortCaption = arguments.short
        #resultsDF = resultsDF.convert_dtypes()
        # If we don't explicitly convert the Technique column to a string, we see problems
        resultsDF['Technique'] = resultsDF['Technique'].astype('string')
        #resultsDF.drop(["Technique"], inplace=True, axis=1)
        resultsDF.drop_duplicates(inplace=True)

        # Drop the columns except the technique and parameters
        whitelist = ['Technique', 'Parameters']
        for column in resultsDF.columns:
            if column not in whitelist and column.upper() != arguments.criteria:
                resultsDF.drop(column, inplace=True, axis=1)
        #resultsDF['Parameters'] = resultsDF['Parameters'].astype('string')
        # Original
        #print(f"{resultsDF.to_latex(longtable=True, index_names=False, index=False, caption=(longCaption, shortCaption), float_format='%.2f', label=arguments.label)}")
        # The max_colwidth option is needed to avoid the row being truncated
        with pd.option_context("max_colwidth", 1000):
            print(f"{resultsDF.to_latex(longtable=True, index_names=False, index=False, caption=(longCaption, shortCaption),  float_format='%.2f', label=arguments.label, column_format='lcl')}")
        print(f"-----------------")
        exit(0)
    else:
        print(f"No result found composed entirely of: {arguments.types}")
        exit(-1)


if arguments.similarity is not None:
    columnNames = ['Technique'] #, 'Accuracy']
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
                    log.debug(f"Technique: {result.technique} AUC: {result.auc} Similarity: {similarityToSelectionTechnique} {result.parameters} vs {list(computed.loc[parameter])}")
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
    print(f"{resultsSubset.to_latex(longtable=True, index_names=False, index=False, caption=(longCaption, shortCaption), label='table:matches-selection', header=columnNames, float_format='%.2f')}")
    print("-- end latex --")
    exit(0)


if arguments.output is not None:
    if arguments.n == -1:
        for result in allResults:
            result.print()
    else:
        # The top Individual results
        topN = [IndividualResult()] * arguments.n
        # For every one of the topN, allow 100 equivalents
        equivalentN = [list()] * arguments.n
        #equivalentN: list[IndividualResult("XXX")] = list()

        # The lowest and highest results seen
        lowestResultInTopN = 0.0
        highestResultInTopN = 0.0
        lowestResultPosition = 0
        highestPositionUsed = 0
        position = 0
        lowestAccuracy = 100.0
        lowestAccuracyPosition = 0

        #results = allResultsForTechnique.results
        for technique in allResults:
            print(f"Technique: {technique.technique}")
            resultNumber = 0
            for result in technique.results:
                resultNumber += 1
                #print(result)
                if result.status == Status.COMPLETED:
                    topN, equivalentN = insertInTopN(result, criteria, topN, equivalentN)
                else:
                    print(f"Something is wrong: results for #{resultNumber} are not complete. Marked as {result} using {technique.technique}")
        dfTopN = pd.DataFrame(topN)
        #print(f"Top {arguments.n} of {len(results)}")
        if arguments.output == FORMAT_CSV:
            # Column headers
            print("sequence,technique,auc,accuracy,base_similarity,parameters", end="")
            for selection in parameters:
                print(f",{'similarity_' + selection}", end="")
            print()
            # Data for topN and each equivalent
            for i in range(len(topN)):
                if topN[i].status != Status.UNCLAIMED:
                    #print(f"{topN[i]} Equivalents: {len(equivalentN[i])}")
                    print(f"\n{i},{topN[i].technique},{topN[i].auc},{topN[i].accuracy},1,{topN[i].parameters}", end='')
                    for selection in parameters:
                        print(f",{similarity(topN[i].parameters, computed.loc[selection])}", end='')
                    for equivalent in equivalentN[i]:
                        print(f"\n{i},{equivalent.technique},{equivalent.auc},{equivalent.accuracy},{topN[i].similarity(equivalent)},{equivalent.parameters}", end='')
                        for selection in parameters:
                            print(f",{similarity(topN[i].parameters, computed.loc[selection])}", end='')
        elif arguments.output == FORMAT_LATEX:
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
            print(f"{dfTopN.to_latex(longtable=True, index_names=False, index=False, caption=(longCaption, shortCaption), label=arguments.label)}")
            print('----- end latex -------')
        elif arguments.output == FORMAT_TEXT:
            print("Text output not supported")
            exit(-1)
        else:
            print(f"Unknown output format: {arguments.output}")
            #print()

    exit(0)