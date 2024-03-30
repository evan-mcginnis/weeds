#
# F E A T U R E  S E L E C T I O N
#
# Needed to prevent errors in type hints in member method parameters that use the enclosing class
from __future__ import annotations

import threading
import math
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections import Counter

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import RFE
#from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
import pickle
from typing import List
import itertools
import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
import os

import constants
from Factors import Factors
from enum import Enum

from numpy import set_printoptions


import random

# Types of analysis
SELECTION_VARIANCE    = "variance"
SELECTION_UNIVARIATE  = "univariate"
SELECTION_RECURSIVE   = "recursive"
SELECTION_PCA         = "pca"
SELECTION_IMPORTANCE  = "importance"
SELECTION_ALL         = "ALL"

class Output(Enum):
    NOTHING = 0
    TEXT = 1
    LATEX = 2
    CSV = 3
    PICKLE = 4

class Criteria(Enum):
    ACCURACY = 0
    AUC = 1

class Selection(ABC):

    def __init__(self, subset=""):
        allFactors = Factors()
        self._rawData = np.ndarray
        self.log = logging.getLogger(__name__)
        # Debuging -- this was the original
        self._columns = allFactors.getColumns([constants.PROPERTY_FACTOR_COLOR, constants.PROPERTY_FACTOR_GLCM])
        #self._columns = allFactors.getColumns([constants.PROPERTY_FACTOR_COLOR, constants.PROPERTY_FACTOR_GLCM, constants.PROPERTY_FACTOR_SHAPE])
        # If we are in the middle of adding a new reading, this creates a bit og a problem,
        # So if a restricted subset is specified, use that -- otherwise, load everything
        #self._columns = allFactors.getColumns(None)
        self._columns.append(constants.NAME_TYPE)
        self._maxFactors = 10
        self._name = ""
        # self._columns = [constants.NAME_RATIO,
        #                  constants.NAME_SHAPE_INDEX,
        #                  constants.NAME_DISTANCE,
        #                  constants.NAME_DISTANCE_NORMALIZED,
        #                  constants.NAME_HUE,
        #                  constants.NAME_SATURATION,
        #                  constants.NAME_I_YIQ,
        #                  constants.NAME_COMPACTNESS,
        #                  constants.NAME_ELONGATION,
        #                  constants.NAME_ECCENTRICITY,
        #                  constants.NAME_ROUNDNESS,
        #                  constants.NAME_SOLIDITY,
        #                  # GLCM
        #                  constants.NAME_HOMOGENEITY,
        #                  constants.NAME_ENERGY,
        #                  constants.NAME_DISSIMILARITY,
        #                  constants.NAME_ASM,
        #                  constants.NAME_CONTRAST,
        #                  constants.NAME_TYPE]

        self._results = pd.DataFrame(columns=self._columns)
        self._uniqueResults = pd.DataFrame()
        self._df = pd.DataFrame()

        return

    @staticmethod
    def supportedSelections():
        return [SELECTION_VARIANCE, SELECTION_UNIVARIATE, SELECTION_RECURSIVE, SELECTION_PCA, SELECTION_IMPORTANCE, SELECTION_ALL]

    @abstractmethod
    def create(self):
        return

    @abstractmethod
    def analyze(self, outputFormat: Output):
        return

    def results(self, unique=False) -> pd.DataFrame:
        return self._results if not unique else self._uniqueResults

    @property
    def name(self) -> str:
        return self._name

    @property
    def maxFactors(self) -> int:
        return self._maxFactors

    @maxFactors.setter
    def maxFactors(self, theMaximum: int):
        self._maxFactors = theMaximum

    @property
    def rawData(self):
        return self._rawData



    def load(self, filename: str):
        # Confirm the file exists
        if not os.path.isfile(filename):
            raise FileNotFoundError(filename)

        # Works
        # self._df = pd.read_csv(filename,
        #                        usecols=[constants.NAME_RATIO,
        #                                 constants.NAME_SHAPE_INDEX,
        #                                 constants.NAME_DISTANCE,
        #                                 constants.NAME_DISTANCE_NORMALIZED,
        #                                 constants.NAME_HUE,
        #                                 constants.NAME_SATURATION,
        #                                 constants.NAME_I_YIQ,
        #                                 constants.NAME_COMPACTNESS,
        #                                 constants.NAME_ELONGATION,
        #                                 constants.NAME_ECCENTRICITY,
        #                                 constants.NAME_ROUNDNESS,
        #                                 constants.NAME_SOLIDITY,
        #                                 # GLCM
        #                                 constants.NAME_HOMOGENEITY,
        #                                 constants.NAME_ENERGY,
        #                                 constants.NAME_DISSIMILARITY,
        #                                 constants.NAME_ASM,
        #                                 constants.NAME_CONTRAST,
        #                                 constants.NAME_TYPE])
        self._df = pd.read_csv(filename, usecols=self._columns)
        self.log.info("Loaded training file")

        # Keep a copy of this -- we will use this elsewhere
        self._rawData = self._df

    def _output(self, format: Output, shortCaption, longCaption: str):
        if format == Output.LATEX:
            self.outputLatex(shortCaption, longCaption)
        elif format == Output.TEXT:
            print(f"{self._results}")

    def outputLatex(self, shortCaption: str, longCaption: str):
        # Transform the frame so we can fit the output on a page
        self._results = self._results.transpose()
        # output the latex data, but I haven't figured out how to set the header
        print(f"{self._results.to_latex(longtable=True, caption= (longCaption, shortCaption), header=['CORRECT THIS'])}")
        # Extract the type -- there should be only two, desired and undesired
        # y = self._df.type
        # self._y = y
        # Drop the type column
        # self._df.drop("type", axis='columns', inplace=True)
        # Drop any data that is not part of the factors we want to consider
        # TODO: Put references to height

# https://www.geeksforgeeks.org/ml-extra-tree-classifier-for-feature-selection/
class FeatureImportance(Selection):
    def __init__(self):
        super().__init__()
        self._name = "Importance"


    def create(self):
        return

    def analyze(self, outputFormat: Output, **kwargs):
        features = self._rawData.values
        # x is everything
        # y is just the type
        x = features[:, 0:self._rawData.shape[1] - 1]
        y = features[:, self._rawData.shape[1] - 1]

        # feature extraction
        model = ExtraTreesClassifier(n_estimators=10, random_state=42)
        model.fit(x, y)
        # print(model.feature_importances_)

        results = {}
        i = 0
        for column in self._df.columns:
            if i < len(model.feature_importances_):
                if outputFormat == Output.TEXT:
                    print(f"{column:20}: {model.feature_importances_[i]}")
                results[column] = model.feature_importances_[i]
                i += 1
            else:
                if outputFormat == Output.TEXT:
                    print(f"{column:20}: ---")
                results[column] = 0

        # The first N items
        sortedResults = OrderedDict(sorted(results.items(), key=lambda t: t[1], reverse=True)[:self._maxFactors])
        self._results = pd.DataFrame([sortedResults])

        self._output(outputFormat, "Feature Importances", "Feature Importances")

        # if outputFormat == Output.LATEX:
        #     self.outputLatex("Feature importances", "Feature Importances")

class PrincipalComponentAnalysis(Selection):
    def __init__(self):
        super().__init__()
        self._name = "PCA"

    def create(self):
        return

    def analyze(self, outputFormat: Output):
        features = self._rawData.values
        # x is everything
        # y is just the type
        x = features[:, 0:self._rawData.shape[1] - 1]
        y = features[:, self._rawData.shape[1] - 1]
        # Originally, the number of components was lower than the number of samples
        #pca = PCA(n_components=len(self._columns) - 1)
        #print(f"PCA: Shape of data {self._rawData.shape} Columns {len(self._columns)} Y {len(y)}")
        pca = PCA(n_components=min(len(self._columns) - 1, len(y)))
        fit = pca.fit(x)
        if outputFormat == Output.TEXT:
            print("Explained Variance: %s" % fit.explained_variance_ratio_)
            print(fit.components_)

        results = {}
        i = 0
        for column in self._df.columns:
            if i < len(fit.explained_variance_ratio_):
                if outputFormat == Output.TEXT:
                    print(f"{column:20}: {fit.explained_variance_ratio_[i]}")
                results[column] = fit.explained_variance_ratio_[i]
                i += 1
            else:
                if outputFormat == Output.TEXT:
                    print(f"{column:20}: ---")
                results[column] = 0

        sortedResults = OrderedDict(sorted(results.items(), key=lambda t: t[1], reverse=True)[:self._maxFactors])
        self._results = pd.DataFrame([sortedResults])

        if outputFormat == Output.LATEX:
            self.outputLatex("PCA", "PCA")

        plt.figure()
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.title("Explained Variance of components")
        plt.xlabel('number of components')
        plt.ylabel('cumulative explained variance');
        plt.savefig("pca.png")
        # plt.show()
        return


class Recursive(Selection):
    def __init__(self):
        super().__init__()
        self._name = "Recursive"

    def create(self):
        return

    def analyze(self, outputFormat: Output):
        features = self._rawData.values
        # x is everything
        # y is just the type
        x = features[:, 0:self._rawData.shape[1] - 1]
        y = features[:, self._rawData.shape[1] - 1]
        weights = {0: 0.8, 1: 0.2}
        # model = LogisticRegression(solver='lbfgs', max_iter=200)
        # model = LogisticRegression(solver='liblinear', class_weight=weights, max_iter=200)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        # Here we use 1 feature so we know the rankings -- if we choose a higher number, they all have that rank
        rfe = RFE(model, n_features_to_select=1)
        fit = rfe.fit(x, y)
        if outputFormat == Output.TEXT:
            print("Num Features: %d" % fit.n_features_)
            print("Selected Features: %s" % fit.support_)
            print("Feature Ranking: %s" % fit.ranking_)

        # Create a dictionary with the results
        results = {}
        i = 0
        for feature in self._columns:
            if i < len(fit.ranking_):
                rank = fit.ranking_[i]
                i += 1
                results[feature] = rank
        sortedResults = OrderedDict(sorted(results.items(), key=lambda t: t[1])[:self._maxFactors])
        self._results = pd.DataFrame([sortedResults])

        if outputFormat == Output.LATEX:
            self.outputLatex("RFE", "Recursive Feature Elimination")


class Variance(Selection):
    def __init__(self):
        super().__init__()
        self._name = "Variance"

    def create(self):
        return

    def analyze(self, outputFormat: Output):
        blobs = self._rawData
        if outputFormat == Output.TEXT:
            print(np.var(blobs, axis=0))
            # np.concatenate((b.reshape(-1, 1), a), axis=1)
        sortedResults = OrderedDict(sorted(blobs, key=lambda t: t[1])[:self._maxFactors])
        self._results = pd.DataFrame([sortedResults])


class Univariate(Selection):


    def __init__(self):
        super().__init__()
        self._name = "Univariate"

    def create(self):
        try:
            self._test = SelectKBest(score_func=f_classif, k="all")
        except UserWarning as r:
            self.log.error(f"{self._name}: {r}")
        return

    def analyze(self, outputFormat: Output, **kwargs) -> np.ndarray:
        # Get the names of the features
        blobs = self._rawData
        names = blobs.columns.values.tolist()
        #self.log.debug("Candidates for feature selection: {}".format(names))

        features = blobs.values
        # x is everything
        # y is just the type
        x = features[:, 0:blobs.shape[1] - 1]
        y = features[:, blobs.shape[1] - 1]

        try:
            fit = self._test.fit(x, y)
        except UserWarning as uWarning:
            self.log.error(f"f{uWarning}")
        except RuntimeWarning as rWarning:
            self.log.error(f"{rWarning}")

        set_printoptions(precision=3, suppress=True, linewidth=120)
        results = {}
        i = 0
        for column in self._df.columns:
            if i < len(fit.scores_):
                if outputFormat == Output.TEXT:
                    print(f"{column:20}: {fit.scores_[i]}")
                results[column] = fit.scores_[i]
                i += 1
            else:
                if outputFormat == Output.TEXT:
                    print(f"{column:20}: ---")
                results[column] = 0

        if outputFormat == Output.TEXT:
            print("Fit scores")
            print(fit.scores_)
            print("Features")
            features = fit.transform(x)
            print(features[0:5, :])

        # Sort the result in descending order, and create a dataframe
        sortedResults = OrderedDict(sorted(results.items(), key=lambda t: t[1], reverse=True)[:self._maxFactors])
        self._results = pd.DataFrame([sortedResults])

        if outputFormat == Output.LATEX:
            self.outputLatex("Univariate", "Univariate parameter selection")
        # # Transform the frame so we can fit the output on a page
        # self._results = self._results.transpose()
        # # output the latex data, but I haven't figured out how to set the header
        # print(f"{self._results.to_latex(longtable=True, caption= 'Univariate', header=['CORRECT THIS'])}")

        return

class All(Selection):
    def __init__(self):
        super().__init__()
        self._variance = Variance()
        self._recursive = Recursive()
        self._pca = PrincipalComponentAnalysis()
        self._importance = FeatureImportance()
        self._univariate = Univariate()
        # TODO: Get variance to work correctly
        #self._selectionTechniques = [self._variance, self._recursive, self._pca, self._importance, self._univariate]
        # This is the original
        #self._selectionTechniques = [self._recursive, self._pca, self._importance, self._univariate]
        # This is the debug list -- recursive is very slow & univariate encounters errors
        self._selectionTechniques = [self._recursive, self._pca, self._importance, self._univariate]


    def create(self):
        """
        Call all the create functions for the techniques
        """
        for technique in self._selectionTechniques:
            self.log.debug(f"Preparing selection technique: {technique.name}")
            technique.maxFactors = self.maxFactors
            technique.create()

    def analyze(self, outputFormat: Output, **kwargs):
        """
        Call all the analyze functions for the techniques
        :param outputFormat:
        """
        if "prefix" in kwargs:
            prefix = kwargs["prefix"]
        else:
            prefix = "parameters"

        consolidated = []
        i = 0
        consolidatedTable = pd.DataFrame(np.nan, index=range(len(self._selectionTechniques)), columns=range(self._maxFactors))
        # Work through all the techniques and store the results
        for technique in self._selectionTechniques:
            technique.analyze(Output.NOTHING)
            factors = technique.results().columns.tolist()
            consolidatedTable.loc[i] = factors
            i += 1
        names = [name.name for name in self._selectionTechniques]
        consolidatedTable.index = names

        # The consolidated results -- each row is a technique with factors in order as columns
        self._results = consolidatedTable
        allFactors = consolidatedTable.stack().values
        self._uniqueResults = np.unique(allFactors)

        if outputFormat == Output.LATEX:
            longCaption = "Parameter Rankings"
            shortCaption = "Parameter Rankings"
            headers = names
            # headers.insert(0, "Rank")
            print(f"{consolidatedTable.T.to_latex(longtable=True, index_names=True, caption=(longCaption, shortCaption), header=headers)}")
            consolidatedTable.to_latex()
        elif outputFormat == Output.CSV:
            consolidatedTable.to_csv('parameters.csv', header=False)
        elif outputFormat == Output.PICKLE:
            consolidatedTable.to_pickle(prefix + ".pickle")


    def load(self, filename: str):
        """
        Load the data for all the techniques
        :param filename: Training data
        """
        for technique in self._selectionTechniques:
            technique.load(filename)

class Criteria(Enum):
    ACCURACY = 0
    AUC = 1



class Status(Enum):
    UNCLAIMED = -1
    IN_PROGRESS = 0
    COMPLETED = 1
class Maximums:
    def __init__(self, filename: str):
        self._filename = filename

        self._maximums = {
            KNNClassifier.name: {RESULT: 0, PARAMETERS: []},
            LogisticRegressionClassifier.name: {RESULT: 0, PARAMETERS: []},
            DecisionTree.name: {RESULT: 0, PARAMETERS: []},
            RandomForest.name: {RESULT: 0, PARAMETERS: []},
            GradientBoosting.name: {RESULT: 0, PARAMETERS: []},
            SuppportVectorMachineClassifier.name: {RESULT: 0, PARAMETERS: []},
            LDA.name: {RESULT: 0, PARAMETERS: []},
            MLP.name: {RESULT: 0, PARAMETERS: []},
            ExtraTrees.name: {RESULT: 0, PARAMETERS: []}
        }


        self._initialized = False

    def read(self):
        with open(self._filename, "rb") as results:
            self._maximums = pickle.load(results)
        self._initialized = True

    def parameters(self, technique: str) -> ():
        results = self._maximums[technique]
        return results[RESULT], results[PARAMETERS]

    def record(self, technique: str, score: float, parameters: []):
        self._maximums[technique] = {RESULT: score, PARAMETERS: parameters}


    def persist(self):
        with open(self._filename, "ab") as results:
            pickle.dump(self._maximums, results)

class IndividualResult:
    def __init__(self, **kwargs):
        # If the combination has been checked yet
        self._checked = Status.UNCLAIMED
        # The result of that check -- will be -1.0 if not yet complete
        self._accuracy = -1.0
        self._auc = -1.0
        # The list of parameters to use
        self._parameters = List[str]
        self._id = -1
        # The number of results to expect
        self._resultsExpected = len(Criteria)

        if "technique" in kwargs:
            self._technique = kwargs["technique"]
        else:
            self._technique = "XXX"

    @property
    def id(self) -> int:
        return self._id

    @id.setter
    def id(self, theId: int):
        if theId < 0:
            raise ValueError("ID must be 0 or greater")
        self._id = theId

    @property
    def technique(self) -> str:
        return self._technique

    @technique.setter
    def technique(self, theTechnique: str):
        self._technique = theTechnique

    @property
    def accuracy(self) -> float:
        return self._accuracy

    @property
    def auc(self) -> float:
        return self._auc

    @property
    def parameters(self) -> []:
        return self._parameters

    @parameters.setter
    def parameters(self, theParameters: []):
        self._parameters = theParameters

    @property
    def status(self) -> Status:
        return self._checked

    def claim(self) -> []:
        self._checked = Status.IN_PROGRESS
        return self._parameters

    def complete(self, criterion: Criteria, result: float):
        """
        Mark the completion for a criteria
        :param criterion: The criteria for a result
        :param result: The value for the criteria
        """
        self._resultsExpected -= 1
        if self._resultsExpected == 0:
            self._checked = Status.COMPLETED
        if criterion == Criteria.ACCURACY:
            self._accuracy = result
        elif criterion == Criteria.AUC:
            self._auc = result

    # Adapted from https://stackoverflow.com/questions/14720324/compute-the-similarity-between-two-lists

    def _cosineSimilarity(self, c1: Counter, c2: Counter) -> float:
        terms = set(c1).union(c2)
        dotprod = sum(c1.get(k, 0) * c2.get(k, 0) for k in terms)
        magA = math.sqrt(sum(c1.get(k, 0) ** 2 for k in terms))
        magB = math.sqrt(sum(c2.get(k, 0) ** 2 for k in terms))
        return dotprod / (magA * magB)

    def similarity(self, target: IndividualResult) -> float:
        return self._cosineSimilarity(Counter(self._parameters), Counter(target.parameters))


    def __str__(self) -> str:
        theString = f"{self._checked}:{self._accuracy}:{self._parameters}"
        return theString

class AllResults:
    def __init__(self, **kwargs):
        """
        The results of checking
        :param theTechnique: Name of the technique (KNN, SVM, etc.)
        """
        if "technique" in kwargs:
            self._technique = kwargs["technique"]
        else:
            self._technique = "XXX"

        self._results = []  # List of results from Result class above
        self._parameters = []           # List of lists -- all combinations of parameters
        self._lastClaimedPosition = 0
        self._batches = 1
        self._lastClaimedPositionInBatch = [0]

    @property
    def batches(self) -> int:
        """
        The number of batches of results
        :return:
        """
        return self._batches

    @batches.setter
    def batches(self, theBatches: int):
        """
        Set the number of batches considered.  Resets the claimed positions to the beginning of the batch
        :param theBatches:
        """
        self._batches = theBatches
        self._lastClaimedPositionInBatch = [0] * theBatches
        for batch in range(theBatches):
            self._lastClaimedPositionInBatch[batch] = batch * int(len(self._results) / theBatches)


    @property
    def results(self) -> []:
        return self._results

    @property
    def parameters(self) -> []:
        return self._parameters

    @parameters.setter
    def parameters(self, theParameters: []):
        self._parameters = theParameters
        id = 0
        # Create the results list
        for parameterList in self._parameters:
            result = IndividualResult(technique=self._technique)
            result.id = id
            result.parameters = parameterList
            self._results.append(result)
            id += 1

    def getNextUnclaimed(self, batch: int) -> IndividualResult:
        """
        Claim the next unclaimed combination
        :return: Individual result or None
        """
        found = False
        combination = None
        if batch not in range(self._batches):
            raise ValueError(f"Batch size given as {batch} must be between 0 and {self._batches}")
        # The lower bound of the range to search
        lowerBound = int(batch * int((len(self._results) / self._batches)))
        upperBound = int(lowerBound + int(len(self._results) / self._batches))
        position = lowerBound
        #print(f"Processing positions between {lowerBound} and {upperBound}")
        while True:
            combination = self._results[self._lastClaimedPositionInBatch[batch]]
            if combination.status == Status.UNCLAIMED:
                self._results[self._lastClaimedPositionInBatch[batch]].claim()
                self._lastClaimedPositionInBatch[batch] = position
                break
            else:
                position += 1
                self._lastClaimedPositionInBatch[batch] += 1
                combination = None
                if self._lastClaimedPositionInBatch[batch] == len(self._results):
                    break

        return combination
    def recordResult(self, combination: IndividualResult, criterion: Criteria, result: float):
        self._results[combination.id].complete(criterion, result)

    def save(self, fileName: str):
        dbfile = open(fileName, 'wb')
        # source, destination
        pickle.dump(self._results, dbfile)
        dbfile.close()

    def load(self, fileName: str):
        dbfile = open(fileName, 'rb')
        self._results = pickle.load(dbfile)
        dbfile.close()


    def print(self):
        print(f"Technique: {self._technique}")
        for result in self._results:
            print(f"Score: {result.accuracy} Status: {result.status} Parameters: {result.parameters}")

if __name__ == "__main__":
    import argparse
    import math
    import sys
    import time
    import logging
    import logging.config
    import itertools
    import more_itertools
    from threading import Thread, Semaphore
    from typing import List
    #    from Logger import Logger
    from Classifier import Classifier, LogisticRegressionClassifier, KNNClassifier, DecisionTree, RandomForest, GradientBoosting, SuppportVectorMachineClassifier, LDA
    from Classifier import MLP, ExtraTrees
    from Classifier import classifierFactory
    from Performance import Performance
    from enum import Enum

    selector = None
    TECHNIQUE = "TECHNIQUE"
    RESULT = "RESULT"
    PARAMETERS = "PARAMETERS"
    MAXIMUM = "maximum"
    l: List[int]

    MAX_PARAMETERS = 10

    STRATEGY_ACCURACY = "accuracy"
    STRATEGY_AUC = "auc"
    allStrategies = [STRATEGY_ACCURACY, STRATEGY_AUC]



    # A single result

    # The maximum values for each technique

    maximumsAccuracy = {
        KNNClassifier.name: {RESULT: 0, PARAMETERS: []},
        LogisticRegressionClassifier.name: {RESULT: 0, PARAMETERS: []},
        DecisionTree.name: {RESULT: 0, PARAMETERS: []},
        RandomForest.name: {RESULT: 0, PARAMETERS: []},
        GradientBoosting.name: {RESULT: 0, PARAMETERS: []},
        SuppportVectorMachineClassifier.name: {RESULT: 0, PARAMETERS: []},
        LDA.name: {RESULT: 0, PARAMETERS: []},
        MLP.name: {RESULT: 0, PARAMETERS: []},
        ExtraTrees.name: {RESULT: 0, PARAMETERS: []}
    }

    maximumsAUC = {
        KNNClassifier.name: {RESULT: 0, PARAMETERS: []},
        LogisticRegressionClassifier.name: {RESULT: 0, PARAMETERS: []},
        DecisionTree.name: {RESULT: 0, PARAMETERS: []},
        RandomForest.name: {RESULT: 0, PARAMETERS: []},
        GradientBoosting.name: {RESULT: 0, PARAMETERS: []},
        SuppportVectorMachineClassifier.name: {RESULT: 0, PARAMETERS: []},
        LDA.name: {RESULT: 0, PARAMETERS: []},
        MLP.name: {RESULT: 0, PARAMETERS: []},
        ExtraTrees.name: {RESULT: 0, PARAMETERS: []}
    }

    maximumsAccuracyEquivalent = {
        KNNClassifier.name: {RESULT: 0, PARAMETERS: []},
        LogisticRegressionClassifier.name: {RESULT: 0, PARAMETERS: []},
        DecisionTree.name: {RESULT: 0, PARAMETERS: []},
        RandomForest.name: {RESULT: 0, PARAMETERS: []},
        GradientBoosting.name: {RESULT: 0, PARAMETERS: []},
        SuppportVectorMachineClassifier.name: {RESULT: 0, PARAMETERS: []},
        LDA.name: {RESULT: 0, PARAMETERS: []},
        MLP.name: {RESULT: 0, PARAMETERS: []},
        ExtraTrees.name: {RESULT: 0, PARAMETERS: []}
    }

    maximumsAUCEquivalent = {
        KNNClassifier.name: {RESULT: 0, PARAMETERS: []},
        LogisticRegressionClassifier.name: {RESULT: 0, PARAMETERS: []},
        DecisionTree.name: {RESULT: 0, PARAMETERS: []},
        RandomForest.name: {RESULT: 0, PARAMETERS: []},
        GradientBoosting.name: {RESULT: 0, PARAMETERS: []},
        SuppportVectorMachineClassifier.name: {RESULT: 0, PARAMETERS: []},
        LDA.name: {RESULT: 0, PARAMETERS: []},
        MLP.name: {RESULT: 0, PARAMETERS: []},
        ExtraTrees.name: {RESULT: 0, PARAMETERS: []}
    }
    resultsSemaphore = Semaphore()

    def recordMaximum(theTechnique: str, theResult: float, theCriteria: Criteria, theParameters: []):
        """
        Record the maximum for a technique
        :param theCriteria:
        :param theTechnique: The name of the technique (KNN, SVM, etc.)
        :param theResult: The float of the result (0..1)
        :param theParameters: An array of the parameters
        """
        if theCriteria == Criteria.ACCURACY:
            results = maximumsAccuracy[theTechnique]
            maximums = maximumsAccuracy
        elif theCriteria == Criteria.AUC:
            results = maximumsAUC[theTechnique]
            maximums = maximumsAUC

        if theResult > results[RESULT]:
            maximums[theTechnique] = {RESULT: theResult, PARAMETERS: theParameters}
            logging.info(f"Global maximum found for {theTechnique}: {theResult} ({theParameters}) vs {results[RESULT]} ({results[PARAMETERS]}")
        else:
            logging.info(f"Local maximum found for {theTechnique}: {theResult} ({theParameters}) vs {results[RESULT]} ({results[PARAMETERS]}")

    def reportMaximums(baseDirectory: str, baseFilename: str):
        """
        Write the accuracy abd AUC maximums to a file
        :param baseFilename:
        :param baseDirectory:
        """
        filename = baseFilename + '.accuracy.txt'
        resultsFilename = os.path.join(baseDirectory, filename)
        with open(resultsFilename, "w") as results:
            for technique, details in maximumsAccuracy.items():
                results.write(f"{technique}:{details[RESULT]}:{details[PARAMETERS]}\n")

        filename = baseFilename + '.auc.txt'
        resultsFilename = os.path.join(baseDirectory, filename)
        with open(resultsFilename, "w") as results:
            for technique, details in maximumsAUC.items():
                results.write(f"{technique}:{details[RESULT]}:{details[PARAMETERS]}\n")

    # def outputMaximums(format: Output):
    #     maximumsDF = pd.DataFrame(maximumsAccuracy)

    def searchForParameters(algorithm: str, dataFile: str, batch: int, basePath: str):
        """

        :param algorithm:
        :param dataFile:
        :param batch:
        :param basePath:
        """
        logger.info(f"Search using {algorithm} in batch {batch}")
        highestClassificationRate = 0.0
        highestAUC = 0.0
        currentCombination = 0
        while True:
            technique = classifierFactory(algorithm)
            resultsSemaphore.acquire(blocking=True)
            combination = allResultsForATechnique[technique.name].getNextUnclaimed(batch)
            resultsSemaphore.release()
            if combination is None:
                break
            currentCombination += 1
            # Perhaps disk I/O is leading to poor performance, so issue a status statement only after 10000 sets
            if currentCombination % 100 == 0:
                logger.info(f"{technique.name}: combination: {currentCombination} parameters: {combination.parameters}")
            technique.selections = combination.parameters
            technique.load(dataFile, stratify=False)
            #technique.correctImbalance()
            technique.createModel(False)

            logger.debug(f"Technique: {technique.name} Combination: {currentCombination} Accuracy: {technique.accuracy()}")
            # Accuracy
            if technique.accuracy() > 0:
                resultsSemaphore.acquire(blocking=True)
                #anClassificationRate = sum(technique.scores) / len(technique.scores)
                meanClassificationRate = technique.accuracy()
                logger.info(f"{technique.name}: combination: {currentCombination} scores: {technique.scores} parameters: {combination.parameters} accuracy: {meanClassificationRate}")
                allResultsForATechnique[technique.name].recordResult(combination, Criteria.ACCURACY, meanClassificationRate)
                allResultsForATechnique[technique.name].save(arguments.prefix + constants.DELIMETER + STRATEGY_ACCURACY + constants.DELIMETER + technique.name + ".pickle")
                resultsSemaphore.release()
            else:
                logger.error(f"Length of scores is zero. Proceeding")
                meanClassificationRate = 0

            if meanClassificationRate > highestClassificationRate:
                highestClassificationRate = meanClassificationRate
                logger.info(f"Found new max for {technique.name} accuracy:{meanClassificationRate} using {combination}")
                recordMaximum(technique.name, meanClassificationRate, Criteria.ACCURACY, combination)
                reportMaximums(basePath, arguments.prefix + constants.DELIMETER + MAXIMUM)
                resultsSemaphore.release()
            elif meanClassificationRate == highestClassificationRate:
                logger.info(f"Found equivalent for {technique.name:} accuracy: {meanClassificationRate} using {combination}")

            # AUC
            if technique.auc > 0:
                logger.info(f"{technique.name}: combination: {currentCombination} scores: {technique.scores} parameters: {combination.parameters} auc: {technique.auc}")
                allResultsForATechnique[technique.name].recordResult(combination, Criteria.AUC, technique.auc)
                allResultsForATechnique[technique.name].save(arguments.prefix + constants.DELIMETER + STRATEGY_AUC + constants.DELIMETER + technique.name + ".pickle")

            if technique.auc > highestAUC:
                highestAUC = technique.auc
                resultsSemaphore.acquire(blocking=True)
                logger.info(f"Found new max for {technique.name} auc: {technique.auc} using {combination}")
                recordMaximum(technique.name, technique.auc, Criteria.AUC, combination)
                reportMaximums(basePath, arguments.prefix + constants.DELIMETER + MAXIMUM)
                resultsSemaphore.release()
            elif technique.auc == highestAUC:
                logger.info(f"Found equivalent for {technique.name} auc: {technique.auc} using {combination}")

            technique.reset()


    def startupLogger(configFile: str, outputFile: str):
        """
        Initializes two logging systems: the image logger and python centric logging.
        :param configFile:
        :param outputDirectory: The output directory for the images
        :return: The image logger instance
        """
        # Confirm the INI exists
        if not os.path.isfile(configFile):
            print("Unable to access logging configuration file {}".format(arguments.logging))
            sys.exit(1)

        # Initialize logging
        logging.config.fileConfig(configFile)
        log = logging.getLogger("jetson")

        return log

    outputChoices = [c.name for c in Output]

    parser = argparse.ArgumentParser("Feature selection")

    parser.add_argument("-df", "--data", action="store", required=True, help="Name of the data in CSV to evaluate")
    parser.add_argument("-fs", "--selection", action="store", required=True, choices=Selection.supportedSelections(), help="Feature selection")
    parser.add_argument("-f", "--factors", type=int, required=False, default=10)
    parser.add_argument("-lg", "--logging", action="store", default="logging.ini", help="Logging configuration file")
    parser.add_argument("-lf", "--logfile", action="store", default="weeds.log", help="Logging output file")
    parser.add_argument("-l", "--latex", action="store_true", required=False, default=False, help="Output latex tables")
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("-o", "--optimal", action="store_true", required=False, default=False, help="Search for optimal parameters")
    actions.add_argument("-co", "--consolidated", action="store_true", required=False, default=False, help="Show consolidated list of parameters")
    actions.add_argument("-t", "--target", action="store", required=False, type=str, help="Write parameter combinations to this file")

    parser.add_argument("-of", "--outputformat", action="store", required=False, default=Output.LATEX.name, choices=outputChoices, help="Output format")
    parser.add_argument("-m", "--maximums", action="store", required=False, help="(Optimal) Maximum result file")
    parser.add_argument("-d", "--debug", action="store_true", required=False, default=False, help="(Optimal) Process a small subset of parameters")
    parser.add_argument("-p", "--prefix", action="store", required=False, default="parameters", help="(Optimal) Prefix for result files (Required for optimal)")
    parser.add_argument("-b", "--batch", action="store", required=False, type=int, default=500000, help="(Optimal) Batch size for parameter search")
    #parser.add_argument("-c", "--chunks", action="store", required=False, type=int, default=1, help="(Optimal) Number of chunks")
    parser.add_argument("-P", "--performance", action="store", type=str, default="performance.csv", help="Name of performance file")

    # selectionCriteria = parser.add_mutually_exclusive_group()
    # selectionCriteria.add_argument("-a", "--auc", action="store_true", default=False, help="Use AUC for scoring")
    # selectionCriteria.add_argument("-s", "--accuracy", action="store_true", default=False, help="Use model accuracy for scoring")

    arguments = parser.parse_args()

    if arguments.optimal:
        if arguments.prefix is None:
            print(f"Searching for optimal parameters requires prefix specified")
            exit(1)

    logger = startupLogger(arguments.logging, arguments.logfile)

    performance = Performance(arguments.performance)
    (performanceOK, performanceDiagnostics) = performance.initialize()
    if not performanceOK:
        print(performanceDiagnostics)
        sys.exit(1)

    if arguments.selection == SELECTION_UNIVARIATE:
        selector = Univariate()
    elif arguments.selection == SELECTION_VARIANCE:
        selector = Variance()
    elif arguments.selection == SELECTION_RECURSIVE:
        selector = Recursive()
    elif arguments.selection == SELECTION_PCA:
        selector = PrincipalComponentAnalysis()
    elif arguments.selection == SELECTION_IMPORTANCE:
        selector = FeatureImportance()
    elif arguments.selection == SELECTION_ALL:
        selector = All()

    selector.maxFactors = arguments.factors
    selector.load(arguments.data)
    selector.create()

    dataDirectory = os.path.dirname(arguments.data)

    if arguments.maximums is not None:
        theMaximums = Maximums(arguments.maximums)
        theMaximums.read()
        theMaximums.persist()
        sys.exit(0)

    # Write out all the combinations to a file for later consumption
    if arguments.target is not None:
        selector.analyze(Output.NOTHING)
        # The list of all the attributes to be analyzed
        results = selector.results(unique=True)
        combinations = itertools.combinations(results, arguments.factors)
        allCombinations = list(combinations)
        logger.info(f"Search space is {len(allCombinations)} combinations from {len(results)} taken {arguments.factors} at a time")
        with open(arguments.target, "w") as parameters:
            for result in allCombinations:
                parameters.write(f"{result}")
                parameters.write(f"\n")
        sys.exit(0)

    # For each technique, find the optimal set of attributes
    if arguments.optimal:
        # TODO: This is a very slow method
        selector.analyze(Output.NOTHING)
        results = []
        if arguments.debug:
            logger.warning("Processing reduced subset")
            results = ["hue", "cb_mean", "hog_mean", "greyscale_homogeneity_90", "compactness"]
            maxParameters = 3
            combinationsPerBatch = 2
        else:
            # The list of all the attributes to be analyzed
            results = selector.results(unique=True)
            maxParameters = arguments.factors
            combinationsPerBatch = arguments.batch


        combinations = itertools.combinations(results, maxParameters)
        chunks = more_itertools.batched(combinations, combinationsPerBatch)

        combinations2 = itertools.combinations(results, maxParameters)
        #chunks2 = more_itertools.batched(combinations2, combinationsPerBatch)
        #allChunks = list(chunks2)
        allCombinations = list(combinations2)
        logger.info(f"Search space is {len(allCombinations)} combinations from {len(results)} taken {maxParameters} at a time")

        # allCombinations = list(combinations)
        # combinations_1, combinations_2 = itertools.tee(allCombinations, 2)


        allTechniques = [RandomForest(), KNNClassifier(), GradientBoosting(), LogisticRegressionClassifier(), DecisionTree(), SuppportVectorMachineClassifier(), LDA(), MLP(), ExtraTrees()]
        #allTechniques = [KNNClassifier(), RandomForest(), GradientBoosting(), LogisticRegressionClassifier(), DecisionTree(), LDA()]
        allTechniquesNames = [x.name for x in allTechniques]

        # Temporary: Create the result files
        allResultsForATechnique = {}

        threads = list()
        threading.stack_size(1024 * 128)
        classifierID = 0
        chunkID = 0
        # Decide how many batches we have
        totalBatches = int(math.ceil(len(allCombinations) / arguments.batch))
        logger.debug(f"Processing {totalBatches} batch(es)")
        for technique in allTechniquesNames:
            logger.debug(f"Technique: {technique}")
            allResultsForATechnique[technique] = AllResults(technique=technique)
            allResultsForATechnique[technique].parameters = allCombinations
            allResultsForATechnique[technique].batches = totalBatches

        # Use all the techniques
        for classifier in allTechniques:
            for chunk in range(totalBatches):
                logger.info(f"Technique-id: {classifier.name}-{classifierID} searching batch {chunk}")
                # For debugging, don't actually launch the threads
                search = Thread(name=classifier.name + constants.DELIMETER + str(chunkID),
                                target=searchForParameters,
                                args=(classifier.name, arguments.data, chunk, dataDirectory,))

                search.daemon = True
                threads.append(search)
                search.start()
                # This is arbitrary but required to avoid errors in startup, it would seem.
                time.sleep(2)
                #searchForParameters(classifier, arguments.data, allCombinations, dataDirectory)
                chunkID += 1
            classifierID += 1

        # Wait for the threads to finish
        logger.info(f"Wait for {len(threads)} threads to finish")
        for x in threads:
            x.join()

        print(f"Maximums reported in: {os.path.join(dataDirectory, arguments.prefix + constants.DELIMETER + MAXIMUM)}")
        if arguments.latex:
            # Accuracy
            longCaption = "Optimal Parameters by Technique (Accuracy)"
            shortCaption = "Optimal Parameters for Accuracy"
            headers = ["Technique", "Parameters"]
            dfMaximums = pd.DataFrame(maximumsAccuracy)

            # The consolidated results -- each row is a technique with factors in order as columns
            rows, cols = (len(allTechniques), maxParameters)
            arr = [[0 for i in range(cols)] for j in range(rows)]

            i = 0
            for technique, result in maximumsAccuracy.items():
                accuracy = result[RESULT]
                if accuracy > 0:
                    #parameters = [parameter for parameter in result[PARAMETERS]]
                    parameters = list(result[PARAMETERS].parameters)
                    print(f"{technique}: Accuracy: {accuracy} {parameters}")
                    parameters.insert(0, accuracy)
                    arr[i] = parameters
                    i += 1
                else:
                    print(f"{technique}: Shows no accuracy. Not normal.")
            dfMaximums = pd.DataFrame(arr)
            print("---------- begin latex ---------------")
            print(f"{dfMaximums.T.to_latex(longtable=True, index_names=False, index=False, caption=(longCaption, shortCaption), label='table:optimal-accuracy', header=allTechniquesNames)}")
            print("---------- end latex ---------------")

            # AUC
            longCaption = "Optimal Parameters by Technique (AUC)"
            shortCaption = "Optimal Parameters for AUC"
            headers = ["Technique", "Parameters"]
            dfMaximums = pd.DataFrame(maximumsAUC)

            # The consolidated results -- each row is a technique with factors in order as columns
            rows, cols = (len(allTechniques), maxParameters)
            arr = [[0 for i in range(cols)] for j in range(rows)]

            i = 0
            for technique, result in maximumsAUC.items():
                accuracy = result[RESULT]
                #parameters = [parameter for parameter in result[PARAMETERS]]
                parameters = list(result[PARAMETERS].parameters)
                print(f"{technique}: AUC: {accuracy} {parameters}")
                parameters.insert(0, accuracy)
                arr[i] = parameters
                i += 1
            dfMaximums = pd.DataFrame(arr)
            print("---------- begin latex ---------------")
            print(f"{dfMaximums.T.to_latex(longtable=True, index_names=False, index=False, caption=(longCaption, shortCaption), label='table:optimal-auc', header=allTechniquesNames)}")
            print("---------- end latex ---------------")

            # Sloppy
            sys.exit(0)


        #reportMaximums(os.path.join(dataDirectory, 'maximums.txt'))

    if arguments.consolidated:
        selector.analyze(Output.NOTHING)
        results = selector.results(unique=True)
        print(f"{results}")
    else:
        selector.analyze(Output[arguments.outputformat], prefix=arguments.prefix)
