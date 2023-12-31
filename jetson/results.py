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

def insertInTopN(candidate: IndividualResult, topN: []) -> []:
    replaced = False
    for i in range(len(topN)):
        #print(f"Compare {candidate.accuracy} vs {topN[i].accuracy}")
        if candidate.accuracy > topN[i].accuracy:
            print(f"Found new max {candidate.accuracy} for position {i}")
            topN[i] = candidate
            break
    return topN

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
            # populate the first N positions
            if position < arguments.n:
                if result.status == Status.COMPLETED:
                    topN[position] = result
                    if result.accuracy < lowestAccuracy:
                        lowestAccuracy = result.accuracy
                        lowestAccuracyPosition = position
                    position += 1
            else:
                if result.status == Status.COMPLETED:
                    topN = insertInTopN(result, topN)
        for result in topN:
            print(result)
