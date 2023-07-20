#
# F E A T U R E  S E L E C T I O N
#
import threading
from abc import ABC, abstractmethod
from collections import OrderedDict

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import RFE
#from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

import itertools
import numpy as np
import pandas as pd
#import logging
import matplotlib.pyplot as plt
import os

import constants
from Factors import Factors
from enum import Enum

from numpy import set_printoptions

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

class Selection(ABC):

    def __init__(self):
        allFactors = Factors()
        self._rawData = np.ndarray
        self.log = logging.getLogger(__name__)
        self._columns = allFactors.getColumns([constants.PROPERTY_FACTOR_COLOR, constants.PROPERTY_FACTOR_GLCM])
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

        self.log.info("Load training file")
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

        # Keep a copy of this -- we will use this elsewhere
        self._rawData = self._df

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


class FeatureImportance(Selection):
    def __init__(self):
        super().__init__()
        self._name = "Importance"


    def create(self):
        return

    def analyze(self, outputFormat: Output):
        features = self._rawData.values
        # x is everything
        # y is just the type
        x = features[:, 0:self._rawData.shape[1] - 1]
        y = features[:, self._rawData.shape[1] - 1]

        # feature extraction
        model = ExtraTreesClassifier(n_estimators=10)
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

        if outputFormat == Output.LATEX:
            self.outputLatex("Feature importances", "Feature Importances")

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
        # pca = PCA(n_components=len(self._columns) - 1)
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
        self._test = SelectKBest(score_func=f_classif, k="all")
        return

    def analyze(self, outputFormat: Output) -> np.ndarray:
        # Get the names of the features
        blobs = self._rawData
        names = blobs.columns.values.tolist()
        self.log.debug("Candidates for feature selection: {}".format(names))

        features = blobs.values
        # x is everything
        # y is just the type
        x = features[:, 0:blobs.shape[1] - 1]
        y = features[:, blobs.shape[1] - 1]

        try:
            fit = self._test.fit(x, y)
        except UserWarning:
            self.log.error("User warning")
        except RuntimeWarning:
            self.log.error("Runtime warning")

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
        self._selectionTechniques = [self._recursive, self._pca, self._importance, self._univariate]


    def create(self):
        """
        Call all the create functions for the techniques
        """
        for technique in self._selectionTechniques:
            technique.maxFactors = self.maxFactors
            technique.create()

    def analyze(self, outputFormat: Output):
        """
        Call all the analyze functions for the techniques
        :param outputFormat:
        """
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

    def load(self, filename: str):
        """
        Load the data for all the techniques
        :param filename: Training data
        """
        for technique in self._selectionTechniques:
            technique.load(filename)


if __name__ == "__main__":
    import argparse
    import yaml
    import sys
    import time
    import logging
    import logging.config
    import itertools
    import more_itertools
    from threading import Thread, Semaphore
#    from Logger import Logger
    from Classifier import Classifier, LogisticRegressionClassifier, KNNClassifier, DecisionTree, RandomForest, GradientBoosting, SuppportVectorMachineClassifier

    selector = None
    TECHNIQUE = "TECHNIQUE"
    RESULT = "RESULT"
    PARAMETERS = "PARAMETERS"

    MAX_PARAMETERS = 10

    # The maximum values for each technique

    maximums = {
        KNNClassifier.name: {RESULT: 0, PARAMETERS: []},
        LogisticRegressionClassifier.name: {RESULT: 0, PARAMETERS: []},
        DecisionTree.name: {RESULT: 0, PARAMETERS: []},
        RandomForest.name: {RESULT: 0, PARAMETERS: []},
        GradientBoosting.name: {RESULT: 0, PARAMETERS: []},
        SuppportVectorMachineClassifier.name: {RESULT: 0, PARAMETERS: []},
    }

    resultsSemaphore = Semaphore()

    def recordMaximum(theTechnique: str, theResult: float, theParameters: []):
        """
        Record the maximum for a technique
        :param theTechnique: The name of the technique (KNN, SVM, etc.)
        :param theResult: The float of the result (0..1)
        :param theParameters: An array of the parameters
        """
        resultsSemaphore.acquire(blocking=True)
        results = maximums[theTechnique]
        if theResult > results[RESULT]:
            maximums[theTechnique] = {RESULT: theResult, PARAMETERS: theParameters}
        else:
            logging.info(f"Local maximum found for {theTechnique}: {theResult} ({theParameters}) vs {results[RESULT]} ({results[PARAMETERS]}")
        resultsSemaphore.release()

    def reportMaximums(resultsFilename: str):
        with open(resultsFilename, "w") as results:
            for technique, details in maximums.items():
                results.write(f"{technique}:{details[RESULT]} {details[PARAMETERS]}\n")

    def searchForParameters(technique: Classifier, dataFile: str, parameters: [], maximums: str):
        """

        :param technique:
        :param dataFile:
        :param parameters:
        """
        logger.debug(f"Search using {technique.name} in {parameters}")
        highestClassificationRate = 0.0
        currentCombination = 0
        for combination in parameters:
            currentCombination += 1
            if currentCombination % 10000 == 0:
                logger.info(f"Processed {currentCombination} combinations for {technique.name}")
            print(f"{technique.name}: {combination}")
            technique.selections = combination
            technique.load(dataFile, stratify=False)
            technique.createModel(False)
            meanClassificationRate = sum(classifier.scores) / len(classifier.scores)
            if meanClassificationRate > highestClassificationRate:
                highestClassificationRate = meanClassificationRate
                logger.info(f"Found new max for {technique.name}:{meanClassificationRate} using {combination}")
                recordMaximum(technique.name, meanClassificationRate, combination)
                reportMaximums(maximums)

    def startupLogger(configFile: str):
        """
        Initializes two logging systems: the image logger and python centric logging.
        :param configFile:
        :param outputDirectory: The output directory for the images
        :return: The image logger instance
        """

        # The command line argument contains the name of the YAML configuration file.

        # Confirm the YAML file exists
        if not os.path.isfile(configFile):
            print("Unable to access logging configuration file {}".format(configFile))
            sys.exit(1)

        # Initialize logging
        with open(arguments.logging, "rt") as f:
            config = yaml.safe_load(f.read())
            logging.config.dictConfig(config)
            theLogger = logging.getLogger(__name__)
        return theLogger


    parser = argparse.ArgumentParser("Feature selection")

    parser.add_argument("-df", "--data", action="store", required=True,
                        help="Name of the data in CSV for use in logistic regression or KNN")
    parser.add_argument("-fs", "--selection", action="store", required=True, choices=Selection.supportedSelections(),
                        help="Feature selection")
    parser.add_argument("-f", "--factors", type=int, required=False, default=10)
    parser.add_argument("-lg", "--logging", action="store", default="info-logging.yaml",
                        help="Logging configuration file")
    parser.add_argument("-l", "--latex", action="store_true", required=False, default=False,
                        help="Output latex tables")
    parser.add_argument("-o", "--optimal", action="store_true", required=False, default=False, help="Search for optimal parameters")
    parser.add_argument("-c", "--consolidated", action="store_true", required=False, default=False, help="Show consolidated list of parameters")
    arguments = parser.parse_args()

    logger = startupLogger(arguments.logging)

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

    # For each technique, find the optimal set of attributes
    if arguments.optimal:
        selector.analyze(Output.NOTHING)
        # The list of all the attributes to be analyzed
        results = selector.results(unique=True)
        # Temporary debugging to cut down on the scope -- replace with above line and set MAX_PARAMETERS to 10
        #results = ["hue", "cb_mean", "hog_mean", "greyscale_homogeneity_90"]
        combinations = itertools.combinations(results, MAX_PARAMETERS)
        chunks = more_itertools.batched(combinations, 1000000)

        # allCombinations = list(combinations)
        # combinations_1, combinations_2 = itertools.tee(allCombinations, 2)


        allTechniques = [RandomForest(), KNNClassifier(), GradientBoosting(), LogisticRegressionClassifier(), DecisionTree(), SuppportVectorMachineClassifier()]

        # print(f"Total Combinations: {len(allCombinations)}")
        #
        # # Determine the number of groups needed.  Use a small value for debugging
        # if len(allCombinations) > 1000:
        #     groups = int(len(allCombinations) / 1000000)
        # else:
        #     groups = 2
        #
        # chunks = more_itertools.batched(combinations_1, groups)

        threads = list()
        threading.stack_size(1024 * 128)
        classifierID = 0
        chunkID = 0
        for chunk in chunks:
            for classifier in allTechniques:
                subset = list(chunk)
                print(f"Technique: {classifier.name}-{classifierID} searching list of length: {len(subset)}")
                # For debugging, don't actually launch the threads
                search = Thread(name=classifier.name + constants.DELIMETER + str(i),
                                target=searchForParameters,
                                args=(classifier, arguments.data, subset, os.path.join(dataDirectory, 'maximums.txt'),))
                search.daemon = True
                threads.append(search)
                search.start()
                # This is arbitrary but required to avoid errors in startup, it would seem.
                time.sleep(2)
                #searchForParameters(classifier, arguments.data, allCombinations)
                classifierID += 1
            chunkID += 1

        for x in threads:
            x.join()

        print(f"Maximums reported in: {os.path.join(dataDirectory, 'maximums.txt')}")
        #reportMaximums(os.path.join(dataDirectory, 'maximums.txt'))

    if arguments.consolidated:
        selector.analyze(Output.NOTHING)
        results = selector.results(unique=True)
        print(f"{results}")
    else:
        selector.analyze(Output.LATEX if arguments.latex else Output.TEXT)
