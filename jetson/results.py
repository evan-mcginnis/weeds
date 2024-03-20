#
# Various operations on results pickle file
#
import argparse
import pickle
import itertools
import more_itertools

from Selection import AllResults
from Selection import IndividualResult
from Selection import Status

def insertInTopN(candidate: IndividualResult, topN: [], equivalentN: []) -> []:
    """
    Insert the candidate into the top N or the equivalent N
    :param candidate:
    :param topN:
    :param equivalentN:
    :return:
    """
    replaced = False
    for i in range(len(topN)):
        #print(f"Compare {candidate.accuracy} vs {topN[i].accuracy}")
        if candidate.accuracy == topN[i].accuracy:
            #print(f"Found equivalent: {candidate.accuracy} for position {i}")
            equivalentN[i].append(candidate)
            break
        if candidate.accuracy > topN[i].accuracy:
            #print(f"Found new max {candidate.accuracy} for position {i}")
            topN[i] = candidate
            equivalentN[i] = []
            break
    return topN, equivalentN

parser = argparse.ArgumentParser("Results file operation")

OPERATION_SHOW = "show"

parser.add_argument("-p", "--pickle", action="store", required=True, help="Pickle file for technique")
parser.add_argument("-o", "--operation", action="store", required=True, choices=[OPERATION_SHOW], help="Operation")
parser.add_argument("-n", "--n", action="store", required=False, default=-1, type=int, help="Number of combinations")


arguments = parser.parse_args()

allResultsForTechnique = AllResults("XXX")
allResultsForTechnique.load(arguments.pickle)

if arguments.operation == OPERATION_SHOW:
    if arguments.n == -1:
        allResultsForTechnique.print()
    else:
        # The top Individual results
        topN = [IndividualResult("XXX")] * arguments.n
        equivalentN = [list()] * arguments.n

        # The lowest and highest results seen
        lowestResultInTopN = 0.0
        highestResultInTopN = 0.0
        lowestResultPosition = 0
        highestPositionUsed = 0
        position = 0
        lowestAccuracy = 100.0
        lowestAccuracyPosition = 0

        results = allResultsForTechnique.results
        for result in results:
            #print(result)
            if result.status == Status.COMPLETED:
                topN, equivalentN = insertInTopN(result, topN, equivalentN)
            else:
                print("Something is wrong: results are not complete")

        print(f"Top {arguments.n} of {len(results)}")
        for i in range(len(topN)):
            if topN[i].status != Status.UNCLAIMED:
                print(f"{topN[i]} Equivalents: {len(equivalentN[i])}")
        # print(f"Equivalents")
        # for equivalent in equivalentN:
        #     for result in equivalent:
        #         print(result)
