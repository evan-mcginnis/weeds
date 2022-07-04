#
# F E A T U R E  S E L E C T I O N
#
from abc import ABC, abstractmethod

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
import os

import constants

from numpy import set_printoptions

class Selection(ABC):

    def __init__(self):
        self._rawData = np.ndarray

        return

    @staticmethod
    def supportedSelections():
        return ["variance", "univariate", "recursive", "pca", "importance"]

    @abstractmethod
    def create(self):
        return

    @abstractmethod
    def analyze(self):
        return

    @property
    def rawData(self):
        return self._rawData

    def load(self, filename: str):
        # Confirm the file exists
        if not os.path.isfile(filename):
            raise FileNotFoundError(filename)

        self.log.info("Load training file")
        self._df = pd.read_csv(filename,
                               usecols=[constants.NAME_RATIO,
                                        constants.NAME_SHAPE_INDEX,
                                        constants.NAME_DISTANCE,
                                        constants.NAME_DISTANCE_NORMALIZED,
                                        constants.NAME_HUE,
                                        constants.NAME_SATURATION,
                                        constants.NAME_I_YIQ,
                                        constants.NAME_COMPACTNESS,
                                        constants.NAME_ELONGATION,
                                        constants.NAME_ECCENTRICITY,
                                        constants.NAME_ROUNDNESS,
                                        constants.NAME_SOLIDITY,
                                        constants.NAME_TYPE])

        # Keep a copy of this -- we will use this elsewhere
        self._rawData = self._df

        # Extract the type -- there should be only two, desired and undesired
        #y = self._df.type
        #self._y = y
        # Drop the type column
        # self._df.drop("type", axis='columns', inplace=True)
        # Drop any data that is not part of the factors we want to consider
        # TODO: Put references to height

class FeatureImportance(Selection):
    def __init__(self):
        self.log = logging.getLogger(__name__)

    def create(self):
        return

    def analyze(self):
        features = self._rawData.values
        # x is everything
        # y is just the type
        x = features[:,0:self._rawData.shape[1]-1]
        y = features[:,self._rawData.shape[1]-1]

        # feature extraction
        model = ExtraTreesClassifier(n_estimators=10)
        model.fit(x, y)
        print(model.feature_importances_)

class PrincipalComponentAnalysis(Selection):
    def __init__(self):
        self.log = logging.getLogger(__name__)

    def create(self):
        return

    def analyze(self):
        features = self._rawData.values
        # x is everything
        # y is just the type
        x = features[:,0:self._rawData.shape[1]-1]
        y = features[:,self._rawData.shape[1]-1]
        pca = PCA(n_components=12)
        fit = pca.fit(x)
        print("Explained Variance: %s" % fit.explained_variance_ratio_)
        print(fit.components_)

        plt.figure()
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.title("Explained Variance of components")
        plt.xlabel('number of components')
        plt.ylabel('cumulative explained variance');
        plt.show()
        return

class Recursive(Selection):
    def __init__(self):
        self.log = logging.getLogger(__name__)

    def create(self):
        return

    def analyze(self):
        features = self._rawData.values
        # x is everything
        # y is just the type
        x = features[:,0:self._rawData.shape[1]-1]
        y = features[:,self._rawData.shape[1]-1]
        weights = {0: 0.8, 1: 0.2}
        #model = LogisticRegression(solver='lbfgs', max_iter=200)
        #model = LogisticRegression(solver='liblinear', class_weight=weights, max_iter=200)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        rfe = RFE(model, n_features_to_select=1)
        fit = rfe.fit(x, y)
        print("Num Features: %d" % fit.n_features_)
        print("Selected Features: %s" % fit.support_)
        print("Feature Ranking: %s" % fit.ranking_)


class Variance(Selection):
    def __init__(self):
        self.log = logging.getLogger(__name__)
        return

    def create(self):
        return

    def analyze(self):

        blobs = self._rawData
        print(np.var(blobs, axis=0))

        return

class Univariate(Selection):

    def __init__(self):
        self.log = logging.getLogger(__name__)
        return

    def create(self):
        self._test = SelectKBest(score_func=f_classif, k="all")
        return

    def analyze(self) -> np.ndarray:

        # Get the names of the features
        blobs = self._rawData
        names = blobs.columns.values.tolist()
        self.log.debug("Candidates for feature selection: {}".format(names))

        features = blobs.values
        # x is everything
        # y is just the type
        x = features[:,0:blobs.shape[1]-1]
        y = features[:,blobs.shape[1]-1]

        fit = self._test.fit(x,y)

        set_printoptions(precision=3, suppress=True, linewidth=120)
        print("Fit scores")
        print(fit.scores_)
        print("Features")
        features = fit.transform(x)
        print(features[0:5,:])

        return

if __name__ == "__main__":
    import argparse
    import yaml
    import sys
    import logging
    import logging.config
    from Logger import Logger

    def startupLogger(configFile: str):
        """
        Initializes two logging systems: the image logger and python centric logging.
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
            #logger = logging.getLogger(__name__)


    parser = argparse.ArgumentParser("Feature selection")

    parser.add_argument("-df", "--data", action="store", required=True, help="Name of the data in CSV for use in logistic regression or KNN")
    parser.add_argument("-fs", "--selection", action="store", required=True, choices = Selection.supportedSelections(), help="Feature selection")
    parser.add_argument("-lg", "--logging", action="store", default="info-logging.yaml", help="Logging configuration file")
    arguments = parser.parse_args()

    startupLogger(arguments.logging)

    if arguments.selection == "univariate":
        selector = Univariate()
    elif arguments.selection == "variance":
        selector = Variance()
    elif arguments.selection == "recursive":
        selector = Recursive()
    elif arguments.selection == "pca":
        selector = PrincipalComponentAnalysis()
    elif arguments.selection == "importance":
        selector = FeatureImportance()

    selector.load(arguments.data)
    selector.create()
    selector.analyze()
