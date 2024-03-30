#
# Various operations on results pickle file
#
from collections import Counter
import math
import glob

import argparse
import pickle
import itertools
import more_itertools
import pandas as pd
import pandas.errors

from Selection import AllResults
from Selection import IndividualResult
from Selection import Status
from Selection import Criteria

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

parser.add_argument("-c", "--computed", action="store", required=True, help="Pickle format file with computed parameters")
parser.add_argument("-cr", "--criteria", action="store", required=False, default="AUC", choices=criteriaChoices, help="Criteria to determine maximum")
parser.add_argument("-f", "--factors", action="store_true", required=False, default=False, help="Show factors")
parser.add_argument("-n", "--n", action="store", required=False, default=-1, type=int, help="Number of combinations")
parser.add_argument("-o", "--output", action="store", required=True, choices=[FORMAT_LATEX, FORMAT_TEXT, FORMAT_CSV], help="Output format")
parser.add_argument("-p", "--prefix", action="store", required=True, help="File prefix")
parser.add_argument("-r", "--results", action="store", required=False, help="Pickle file for technique")
parser.add_argument("-s", "--similarity", action="store", required=False, type=float, help="Show items where similarity matches")

arguments = parser.parse_args()

criteria = Criteria[arguments.criteria]

# Find data files for each technique
files = []
if arguments.prefix is not None:
    files = glob.glob(arguments.prefix + '_*.pickle')
    if len(files) == 0:
        print(f"Unable to find any result files with prefix {arguments.prefix}")
elif arguments.results is not None:
    files = arguments.results
else:
    print("Must specify either --results or --prefix for result files")

# Load the results from all the <prefix>_*.pickle files, or just the one the user specified
allResults = []
for file in files:
    allResultsForTechnique = AllResults()
    allResultsForTechnique.load(file)
    allResults.append(allResultsForTechnique)

# The calculated parameters
try:
    computed = pd.read_pickle(arguments.computed)
    if arguments.factors:
        print("----- F A C T O R S -----")
        print(f"{computed}")
        print("----- F A C T O R S -----")
        exit(0)
except FileNotFoundError:
    print(f"Unable to access parameter file: {arguments.computed}")
    exit(-1)

# The names of the selection techniques
parameters = computed.index

# if arguments.output is not None:
#     columnNames = ['Technique'] #, 'Accuracy']
#     columnNames.extend(list(parameters))
#     resultsSubset = pd.DataFrame(columns=columnNames)
#     resultRows = []
#     for technique in allResults:
#         for result in technique.results:
#             if result.status == Status.COMPLETED:
#                 row = [result.technique, result.accuracy]
#                 matched = False
#                 for parameter in parameters:
#                     similarityToSelectionTechnique = similarity(result.parameters, computed.loc[parameter])
#                     print(f"Technique: {result.technique} AUC: {result.auc} Similarity: {similarityToSelectionTechnique} {result.parameters} vs {list(computed.loc[parameter])}")
#                     if similarityToSelectionTechnique >= arguments.similarity:
#                         matched = True
#                     row.append(similarity(result.parameters, computed.loc[parameter]))
#
#                 if matched:
#                     resultRows.append(row)
#                     resultsSubset.loc[len(resultsSubset)] = row
#
#     print("-- begin latex --")
#     longCaption = "Parameter selection techniques"
#     shortCaption = "short caption"
#     print(f"{resultsSubset.to_latex(longtable=True, index_names=False, index=False, caption=(longCaption, shortCaption), label='table:matches-selection', header=columnNames, float_format='%.2f')}")
#     print("-- end latex --")
#     exit(0)



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
                    print(f"Technique: {result.technique} AUC: {result.auc} Similarity: {similarityToSelectionTechnique} {result.parameters} vs {list(computed.loc[parameter])}")
                    if similarityToSelectionTechnique >= arguments.similarity:
                        matched = True
                    row.append(similarity(result.parameters, computed.loc[parameter]))

                if matched:
                    resultRows.append(row)
                    resultsSubset.loc[len(resultsSubset)] = row

    print("-- begin latex --")
    longCaption = "Parameter selection techniques"
    shortCaption = "short caption"
    print(f"{resultsSubset.to_latex(longtable=True, index_names=False, index=False, caption=(longCaption, shortCaption), label='table:matches-selection', header=columnNames, float_format='%.2f')}")
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
            #print(result)
            for result in technique.results:
                if result.status == Status.COMPLETED:
                    topN, equivalentN = insertInTopN(result, criteria, topN, equivalentN)
                else:
                    print("Something is wrong: results are not complete")

        #print(f"Top {arguments.n} of {len(results)}")
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
                    #print()
        # print(f"Equivalents")
        # for equivalent in equivalentN:
        #     for result in equivalent:
        #         print(result)
