#
# C L A S S I F I E R
#
import csv
import random

#import random

import constants
from constants import Score
#import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import logging
import warnings
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.neural_network import MLPClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.exceptions import ConvergenceWarning

from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE, KMeansSMOTE, SVMSMOTE
from WeedExceptions import ProcessingError
from abc import ABC, abstractmethod

import os.path
from enum import Enum

class Subset(Enum):
    TRAIN = 0
    TEST = 1
    ALL = 2
    NONE = 3

class Type(Enum):
    CROP = 0
    WEED = 1

class ImbalanceCorrection(Enum):
    SMOTE = 0
    ADASYN = 1
    BORDERLINE = 2
    KMEANS = 3
    SVM = 4

    def __str__(self):
        return self.name

class ClassificationTechniques(Enum):
    KNN = 0
    RANDOMFOREST = 1
    DECISIONTREE = 2
    SVM = 3
    LDA = 4
    MLP = 5
    LOGISTIC = 6
    GRADIENT = 7

    def __str__(self):
        return self.name

class Classifier:
    name = "Base"

    def __init__(self):

        self._model = None
        self._df = pd.DataFrame()
        self._rawData = pd.DataFrame()

        self._xTrain = []
        self._yTrain = pd.Series
        self._xTest = []
        self._yTest = []
        # The actual type of the blob
        self._actual = []

        # Scoring
        self._scoring = Score.UNDEFINED

        # ROC items
        self._fpr = 0.0
        self._tpr = 0.0
        self._auc = 0.0
        self._y_scores = []
        self._threshold = 0.0


        self._blobsInView = pd.DataFrame()
        self._selections = []
        self._scores = []
        self._name = "Base"
        self._loaded = False

        self._correctImbalance = False
        self._writeDatasetToDisk = False
        self._correctImbalanceAlgorithm = ImbalanceCorrection.SMOTE
        self._desiredImbalanceRatio = 1.0
        self._correctSubset = Subset.TRAIN

        self.log = logging.getLogger(__name__)
        # # The ANN for the classifier
        # self._ann = cv.ml.ANN_MLP_create()
        # self._ann.setLayerSizes(np.array([3, 8, 3], np.uint8))
        # # Symetrical sigmoid activation
        # self._ann.setActivationFunction(cv.ml.ANN_MLP_SIGMOID_SYM, 0.6, 1.0)
        # # Backpropogation
        # self._ann.setTrainMethod(cv.ml.ANN_MLP_BACKPROP, 0.1, 0.1)
        # self._ann.setTermCriteria((cv.TERM_CRITERIA_MAX_ITER | cv.TERM_CRITERIA_EPS, 100, 1.0))
        self._scaler = StandardScaler()
        return

    # ANN Support routines start
    # def _record(self, sample: [], classification: []):
    #     return (np.array([sample], np.float32), np.array([classification], np.float32))
    #
    # def _vectorized(self, i: int) -> []:
    #     """
    #     A utility function to convert a value into an array where the ith value is 1.0
    #     :param i:
    #     :return:
    #     Numpy array with the ith element as 1.0
    #     """
    #     #e = np.zeros((4,), np.float32)
    #     e = [0.0, 0.0, 0.0, 0.0]
    #     e[i] = 1.0
    #     return e
    #
    # def _constructData(self):
    #     """
    #     Construct the data in the form opencv wants it
    #     """
    #     self._records = []
    #     for blobName, blobAttributes in self._blobs.items():
    #         self._records.append(self._record([blobAttributes[constants.NAME_AREA],
    #                                            blobAttributes[constants.NAME_RATIO],
    #                                            blobAttributes[constants.NAME_RATIO]],
    #                                           self._vectorized(blobAttributes[constants.NAME_TYPE])))
    # def train(self, blobs: {}):
    #     EPOCHS = 10
    #
    #     self._constructData()
    #
    #     for e in range(0, EPOCHS):
    #         print("Train %d\n" % e)
    #         for (t, c) in self._records:
    #             data = cv.ml.TrainData_create(t, cv.ml.ROW_SAMPLE, c)
    #             if self._ann.isTrained():
    #                 self._ann.train(data,
    #                                 cv.ml.ANN_MLP_UPDATE_WEIGHTS,
    #                                 cv.ml.ANN_MLP_NO_INPUT_SCALE,
    #                                 cv.ml.ANN_MLP_NO_OUTPUT_SCALE)
    #             else:
    #                 self._ann.train(data, cv.ml.ANN_MLP_NO_INPUT_SCALE, cv.ml.ANN_MLP_NO_OUTPUT_SCALE)
    #     return
    # ANN support routines end

    @classmethod
    def correctionAlgorithms(cls) -> []:
        """
        Supported algorithms for imbalance correction
        :return: list of names
        """
        choices = []
        for choice in ImbalanceCorrection:
            choices.append(choice.name)
        return choices

    @property
    def correctSubset(self) -> Subset:
        return self._correctSubset

    @correctSubset.setter
    def correctSubset(self, theSubset: Subset):
        self._correctSubset = theSubset

    @property
    def writeDatasetToDisk(self) -> bool:
        return self._writeDatasetToDisk

    @writeDatasetToDisk.setter
    def writeDatasetToDisk(self, writeData: bool):
        self._writeDatasetToDisk = writeData

    @property
    def minority(self) -> float:
        return self._desiredImbalanceRatio

    @minority.setter
    def minority(self, desiredMinority: float):
        if desiredMinority < 0.0 or desiredMinority > 1.0:
            raise AttributeError(f"Minority {desiredMinority} not within range 0..1")
        self._desiredImbalanceRatio = desiredMinority

    @property
    def correct(self) -> bool:
        """
        Correct imbalances in dataset
        :return:
        """
        return self._correctImbalance

    @correct.setter
    def correct(self, theCorrection: bool):
        self._correctImbalance = theCorrection

    @property
    def correctionAlgorithm(self) -> ImbalanceCorrection:
        """
        The algorithm used in correction
        :return:
        """
        return self._correctImbalanceAlgorithm

    @correctionAlgorithm.setter
    def correctionAlgorithm(self, theAlgorithm: ImbalanceCorrection):
        self._correctImbalanceAlgorithm = theAlgorithm

    def createImbalance(self, percentage: float, location: Subset):
        """
        Create an imbalance in the data set -- should be called before a split
        :param location: Create imbalance in train, test, or all
        :param percentage: percentage between 0..1 of the minority class that remains
        """
        self.log.debug(f"Imbalance before creation: {self.imbalanceRatio(location)}")
        if location == Subset.ALL:
            candidates = self._df.index[self._df[constants.NAME_TYPE] == 0].tolist()
            indicesToDrop = []
            random.seed(42)
            for index in candidates:
                if random.random() > percentage:
                    indicesToDrop.append(index)
            self._df.drop(index=indicesToDrop, inplace=True)
            self.log.debug(f"Created imbalance in {location.name} by dropping {indicesToDrop}")
        elif location == Subset.TRAIN:
            candidates = self._yTrain.index[self._yTrain == 1].tolist()
            indicesToDrop = []
            random.seed(42)
            for index in candidates:
                if random.random() > percentage:
                    indicesToDrop.append(index)
            self._xTrain.drop(index=indicesToDrop, inplace=True)
            self._yTrain.drop(index=indicesToDrop, inplace=True)
            self.log.debug(f"Created imbalance in {location.name} by dropping {indicesToDrop}")
        elif location == Subset.NONE:
            return
        else:
            self.log.error(f"Unable to create imbalance in subset of data")
        self.log.debug(f"Imbalance after: {self.imbalanceRatio(location)}")

    def correctImbalance(self, location: Subset = Subset.NONE):
        """
        Correct imbalance between majority and minority classes in dataset.
        This updates the training dataset.
        Should be called only after the training split.
        """
        # Follow the explanation here:
        # https://stackoverflow.com/questions/15065833/imbalance-in-scikit-learn
        # if not self._loaded:
        #     raise ProcessingError("Data must be loaded before imbalances can be corrected")

        # debug the correction
        # df = pd.DataFrame(self._xTrain)
        # df.to_csv("before-imbalance-correction.csv")

        self.log.debug(f"Correcting imbalance in {location.name} using {self._correctImbalanceAlgorithm.name}  Currently {self.imbalanceRatio(location)}")

        if self._correctImbalanceAlgorithm == ImbalanceCorrection.SMOTE:
            corrector = SMOTE(random_state=2)
        elif self._correctImbalanceAlgorithm == ImbalanceCorrection.ADASYN:
            corrector = ADASYN(random_state=2)
        elif self._correctImbalanceAlgorithm == ImbalanceCorrection.BORDERLINE:
            corrector = BorderlineSMOTE(random_state=2)
        elif self._correctImbalanceAlgorithm == ImbalanceCorrection.KMEANS:
            corrector = KMeansSMOTE(random_state=2)
        elif self._correctImbalanceAlgorithm == ImbalanceCorrection.SVM:
            corrector = SVMSMOTE(random_state=2)
        else:
            raise AttributeError(f"Requested algorithm not supported: {self._correctImbalanceAlgorithm.name}")


        if location == Subset.TRAIN:
            self._xTrain, self._yTrain = corrector.fit_resample(self._xTrain, self._yTrain)
        elif location == Subset.ALL:
            self._x, self._y = corrector.fit_resample(self._x, self._y)
        elif location == Subset.TEST:
            raise NotImplementedError("Unable to correct test subset")
        elif location == Subset.NONE:
            self.log.debug("No subset selected for correction")
        else:
            raise AttributeError(f"Unknown location {location}")

        self.log.debug(f"Corrected imbalance.  Currently {self.imbalanceRatio(location)}")


        # debug the correction
        # df = pd.DataFrame(self._xTrain)
        # df.to_csv("after-imbalance-correction.csv")


    def imbalanceRatio(self, location: Subset) -> str:
        """
        The imbalance ratio between the majority and minority classes
        :return:
        """
        # if not self._loaded:
        #     raise ProcessingError("Data must be loaded before imbalances can be corrected")

        unique = []
        counts = []

        if location == Subset.TRAIN:
            counts = self._yTrain.value_counts()
        elif location == Subset.ALL:
            counts = self._y.value_counts()
        elif location == Subset.TEST:
            # The counts of the classes
            unique, counts = np.unique(self._yTest, return_counts=True)
        elif location == Subset.NONE:
            return "0"
        else:
            raise NotImplementedError(f"Unknown location: {location}")

        #self.log.debug(f"Class counts: {counts}")
        # There should be only 2 counts for the two classes
        assert len(counts) == 2

        ratio = str(counts[0]) + ':' + str(counts[1]) + ' ' + Type(0).name + ':' + Type(1).name + ' (' + str(counts[0] / counts[1]) + ')'

        return ratio

    @property
    def model(self):
        return self._model

    @property
    def scoring(self) -> Score:
        """
        The current scoring model of the classifier
        :return: Score
        """
        return self._scoring

    @scoring.setter
    def scoring(self, theScoring: Score):
        """
        Set the scoring model for the classifier
        :param theScoring:
        """
        self._scoring = theScoring

    def accuracy(self) -> float:
        self._model.predict(self._xTest)
        return self._model.score(self._xTest, self._yTest)

    def averageOfCrossValidation(self) -> float:
        """
        The average of the model cross validations
        :return:
        """
        return sum(self._scores) / len(self._scores)

    @property
    def tpr(self) -> float:
        """
        The true positive rate
        :return:
        """
        return self._tpr

    @property
    def fpr(self) -> float:
        """
        The false positive rate
        :return:
        """
        return self._fpr

    @property
    def auc(self) -> float:
        return self._auc

    @property
    def actual(self) -> []:
        return self._actual

    @property
    def scores(self) -> []:
        return self._scores

    @property
    def blobs(self):
        return self._blobs

    @blobs.setter
    def blobs(self, blobSet: {}):
        self._blobs = blobSet

    @property
    def rawData(self) -> pd.DataFrame:
        return self._rawData

    def classifyWithinCropRow(self):
        return

    def classifyByDamage(self, rectangles: []):
        self.log.disabled = False
        for rectName, rectAttributes in rectangles.items():
            (x, y, w, h) = rectAttributes[constants.NAME_LOCATION]
            (cX, cY) = rectAttributes[constants.NAME_CENTER]
            self.log.debug("Bounding: ({},{}) {}x{} Center ({},{})".format(x, y, w, h, cX, cY))

    #
    # The only real purpose here is to mark the items that are at the edge of the image as such
    #
    def classifyByPosition(self, size : ()):
        """
        Classify items as unknown if they extend off the edge of the image.
        :param size: Unused. Should be refactored out.
        """
        self.log.info("Classify by position")
        (maxY, maxX, depth) = size
        for blobName, blobAttributes in self._blobs.items():
            (x, y, w, h) = blobAttributes.get(constants.NAME_LOCATION)
            if (x == 0 or x + w >= maxX):
                blobAttributes[constants.NAME_TYPE] = constants.TYPE_UNKNOWN
                blobAttributes[constants.NAME_REASON] = constants.REASON_AT_EDGE

    def classifyByRatio(self, largest: int, size : (),ratio: int):
        """
        Classify blobs in the image my size ratio.
        :param size: size of the images
        :param largest: the size in pixels of the blob with the largest area
        :param ratio: the threshold ratio
        """
        (maxY, maxX, depth) = size
        try:
            largestArea = self._blobs[largest].get(constants.NAME_AREA)
            for blobName, blobAttributes in self._blobs.items():

                # Only for items that have not already been classified

                if blobAttributes[constants.NAME_TYPE] == constants.TYPE_UNKNOWN:
                    # If the area of the blob is much smaller than the largest object, it must be undesirable
                    if largestArea > ratio * blobAttributes.get(constants.NAME_AREA):

                        # But, if the view of this is only partial in that it is at the edge of the image,
                        # we can't say with confidence that it is
                        (x, y, w, h) = blobAttributes.get(constants.NAME_LOCATION)
                        if(x == 0 or x+w >= maxX):
                            blobAttributes[constants.NAME_TYPE] = constants.TYPE_UNKNOWN
                            blobAttributes[constants.NAME_REASON] = constants.REASON_AT_EDGE
                        else:
                            blobAttributes[constants.NAME_TYPE] = constants.TYPE_UNDESIRED
                            blobAttributes[constants.NAME_REASON] = constants.REASON_SIZE_RATIO
                    else:
                        (x, y, w, h) = blobAttributes.get(constants.NAME_LOCATION)
                        if(x == 0 or x+w >= maxX):
                            blobAttributes[constants.NAME_TYPE] = constants.TYPE_UNKNOWN
                            blobAttributes[constants.NAME_REASON] = constants.REASON_AT_EDGE
                        else:
                            blobAttributes[constants.NAME_TYPE] = constants.TYPE_DESIRED
                            blobAttributes[constants.NAME_REASON] = constants.REASON_SIZE_RATIO
        except KeyError:
            return

    def classify(self, reason: int):
        self._prepareData()

        self.log.info("Classify")
        predictions = self._classifier.predict(self._blobsInView)
                # Mark up the current view
        i = 0
        for blobName, blobAttributes in self._blobs.items():
            if blobAttributes[constants.NAME_REASON] != constants.REASON_AT_EDGE:
                blobAttributes[constants.NAME_TYPE] = predictions[i]
                blobAttributes[constants.NAME_REASON] = reason
            i = i + 1
        return

    def loadSelections(self, filename: str):
        """
        Read in the selections from a CSV file with only one line.
        :param filename:
        :return:
        """

        # This is no longer needed, as we get the selections from the INI file
        raise NotImplementedError

        # if not os.path.isfile(filename):
        #     raise FileNotFoundError
        #
        # self.log.info("Load parameter selections")
        # # The selection file is just a single line of CSV
        # with open(filename) as f:
        #     reader = csv.reader(f)
        #     row = next(reader)
        # self._selections = row

        # return True

    @property
    def selections(self) -> []:
        return self._selections

    @selections.setter
    def selections(self, selectionList: []):
        self._selections = selectionList

    def load(self, filename: str, stratify: bool):
        # Confirm the file exists
        if not os.path.isfile(filename):
            self.log.critical("Unable to load file: {}".format(filename))
            raise FileNotFoundError

        if self._loaded:
            self.log.info(f"Data is already loaded.")
            return

        self.log.info("Load training file")

        # This reads the hard-coded selectioni (this works)
        # self._df = pd.read_csv(filename,
        #                        usecols=[constants.NAME_RATIO,
        #                                 constants.NAME_SHAPE_INDEX,
        #                                 constants.NAME_DISTANCE,
        #                                 constants.NAME_DISTANCE_NORMALIZED,
        #                                 constants.NAME_HUE,
        #                                 constants.NAME_I_YIQ,
        #                                 constants.NAME_TYPE])

        # Get the type as well as everything in the selection
        s = list(self._selections)
        s.append(constants.NAME_TYPE)
        self.log.debug("Using columns: {}".format(s))
        self._df = pd.read_csv(filename, usecols=s)

        # Keep a copy of this -- we will use this elsewhere
        self._rawData = self._df.copy(deep=True)

        # Extract the type -- there should be only two, desired and undesired
        y = self._df.type
        self._y = y
        # Drop the type column
        self._df.drop("type", axis='columns', inplace=True)
        # Sort by the column names, so the train columns and the features extracted match.
        # Mixing up the order does not cause a failure in scikit-learn, but this is required.
        self._df.sort_index(axis=1, inplace=True)

        self._x = self._df
        # Drop any data that is not part of the factors we want to consider

        #self.log.debug(f"Entire dataset imbalance before correction: {self.imbalanceRatio(Subset.ALL)}")

        if self._writeDatasetToDisk:
            df = pd.DataFrame(self._xTrain, columns=s)
            df.type = self._yTrain
            df.to_csv(f"before-{self._correctImbalanceAlgorithm.name.lower()}-{self._desiredImbalanceRatio:.2f}-correction.csv")

        # If we want the entire dataset corrected, do so before the split into train and test
        if self._correctImbalance and self._correctSubset == Subset.ALL:
            self.correctImbalance(Subset.ALL)

        # Split up the data
        if stratify:
            X_train, X_test, y_train, y_test = train_test_split(self._df,y,train_size=0.4, stratify=y,random_state=42)
        else:
            X_train, X_test, y_train, y_test = train_test_split(self._df,y,train_size=0.4,random_state=42)
        self._xTrain = X_train
        self._yTrain = y_train
        self._xTest = X_test
        self._yTest = y_test

        # Create the imbalance before anything else is done
        if self._desiredImbalanceRatio != 1.0:
            self.createImbalance(self._desiredImbalanceRatio, Subset.TRAIN)


        # If we are correcting just the train portion, that should be done after the split
        if self._correctImbalance and self._correctSubset == Subset.TRAIN:
            self.correctImbalance(Subset.TRAIN)

        if self._writeDatasetToDisk:
            df = pd.DataFrame(self._xTrain, columns=s)
            df.type = self._yTrain
            df.to_csv(f"after-{self._correctImbalanceAlgorithm.name.lower()}-{self._desiredImbalanceRatio:.2f}-correction.csv")

        self._loaded = True

    def _prepareData(self ):

        features = []
        _features = []
        Xfeatures = []
        # This is the list of features we have chosen to use
        selectedFeatureNames = tuple(self._selections)


        # Build up the list of features we will use.
        # Reading some performance comparisons, this is the fastest way to create a dataframe
        for blobName, blobAttributes in self._blobs.items():

            # Build up a list of features in the same order as the names
            _features = []
            for feature in self._selections:
                _features.append(blobAttributes[feature])

            # Put these in a format so we can initialize the dataframe
            features.append(_features)
            # This is the original, hard-coded version before I transitioned to using just color and GLCM
            # 23 May 2023 this works just fine when I use all the attributes
            # Xfeatures.append([blobAttributes[constants.NAME_RATIO],
            #                  blobAttributes[constants.NAME_SHAPE_INDEX],
            #                  blobAttributes[constants.NAME_DISTANCE],
            #                  blobAttributes[constants.NAME_DISTANCE_NORMALIZED],
            #                  blobAttributes[constants.NAME_HUE],
            #                  blobAttributes[constants.NAME_I_YIQ]])
            # This is the Color+GLCM version 23 May 2023
            # 27 Feb 2024 Not needed
            # Xfeatures.append([blobAttributes[constants.NAME_SATURATION],
            #                  blobAttributes[constants.NAME_I_YIQ],
            #                  blobAttributes[constants.NAME_BLUE_DIFFERENCE],
            #                  blobAttributes[constants.NAME_GREYSCALE + constants.DELIMETER + constants.NAME_HOMOGENEITY + constants.DELIMETER + "0"],
            #                  blobAttributes[constants.NAME_GREYSCALE + constants.DELIMETER + constants.NAME_ENERGY + constants.DELIMETER + "0"],
            #                  blobAttributes[constants.NAME_GREYSCALE + constants.DELIMETER + constants.NAME_CONTRAST + constants.DELIMETER + "0"],
            #                  blobAttributes[constants.NAME_GREYSCALE + constants.DELIMETER + constants.NAME_DISSIMILARITY + constants.DELIMETER + "0"],
            #                  blobAttributes[constants.NAME_GREYSCALE + constants.DELIMETER + constants.NAME_ASM + constants.DELIMETER + "0"]])

            # Construct the dataframe we will use
            #self._blobsInView = pd.DataFrame(features, columns=('ratio', 'shape', 'distance','normalized_distance', 'hue'))
            # 23 May 2023 the original with shape attributes
            # self._blobsInView = pd.DataFrame(Xfeatures, columns=(constants.NAME_RATIO,
            #                                                     constants.NAME_SHAPE_INDEX, #'shape',
            #                                                     constants.NAME_DISTANCE, # 'distance',
            #                                                     constants.NAME_DISTANCE_NORMALIZED, # 'normalized_distance',
            #                                                     constants.NAME_HUE, # 'hue'
            #                                                     constants.NAME_I_YIQ)) # I std deviation

            self._blobsInView = pd.DataFrame(Xfeatures, columns=(constants.NAME_SATURATION,
                                                                 constants.NAME_I_YIQ,
                                                                 constants.NAME_BLUE_DIFFERENCE,
                                                                 constants.NAME_HOMOGENEITY,
                                                                 constants.NAME_ENERGY,
                                                                 constants.NAME_CONTRAST,
                                                                 constants.NAME_DISSIMILARITY,
                                                                 constants.NAME_ASM))
            # Create a dataframe from the with the column names we want and the feature values
            self._blobsInView = pd.DataFrame(features, columns=selectedFeatureNames)
            self._blobsInView.sort_index(axis=1, inplace=True)

    def visualize(self):
        plt.title(f'{self.name} Receiver Operating Characteristic')
        plt.plot(self._fpr, self._tpr, 'b', label='AUC = %0.2f' % self._auc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.title(f'ROC Curve of {self.name}')
        plt.show()

    def visualizeFolds(self):
        # Adapted from:
        # https://stackoverflow.com/questions/57708023/plotting-the-roc-curve-of-k-fold-cross-validation

        cv = StratifiedKFold(n_splits=5)
        #classifier = SVC(kernel='sigmoid', probability=True, random_state=0)

        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        plt.figure(figsize=(10, 10))
        i = 0
        # Original
        #for train, test in cv.split(X_train_res, y_train_res):
        #self._xTrain.reset_index(inplace=True)
        #self._yTrain = self._yTrain.reset_index()
        for train, test in cv.split(self._x, self._y):
            # Original
            # probas_ = classifier.fit(X_train_res[train], y_train_res[train]).predict_proba(X_train_res[test])
            probas_ = self._model.fit(self._x.iloc[train], self._y.iloc[train]).predict_proba(self._x.iloc[test])
            # Compute ROC curve and area the curve
            fpr, tpr, thresholds = roc_curve(self._y[test], probas_[:, 1])
            tprs.append(np.interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            plt.plot(fpr, tpr, lw=1, alpha=0.3,
                     label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

            i += 1
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                 label='Chance', alpha=.8)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        plt.plot(mean_fpr, mean_tpr, color='b',
                 label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                 lw=2, alpha=.8)

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                         label=r'$\pm$ 1 std. dev.')

        plt.xlim([-0.01, 1.01])
        plt.ylim([-0.01, 1.01])
        plt.xlabel('False Positive Rate', fontsize=18)
        plt.ylabel('True Positive Rate', fontsize=18)
        plt.title(f'Cross-Validation ROC of {self.name}', fontsize=18)
        plt.legend(loc="lower right", prop={'size': 15})
        plt.show()

    def createModel(self, score: bool):
        raise NotImplementedError

    def reset(self):
        self._model = None

class MLP(Classifier):
    name = ClassificationTechniques.MLP.name

    def __init__(self):
        super().__init__()
        self._xTrainScaled = None
        self._xTestScaled = None

    def createModel(self, score: Score):
        # Scale the data
        xTrainAsList = self._xTrain.values.tolist()
        self._scaler.fit(xTrainAsList)
        #self._scaler.fit(self._xTrain)
        self._xTrainScaled = self._scaler.transform(xTrainAsList)
        # Keeping this as a numpy array is causing problems
        #scaledXTrain = self._scaler.transform(self._xTrain)
        xTestAsList = self._xTest.values.tolist()
        self._xTestScaled = self._scaler.transform(xTestAsList)

        # The lbfgs solver runs into problems
        self._model = MLPClassifier(solver='adam', max_iter=5000, alpha=1e-5, hidden_layer_sizes=(20), random_state=1)
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                # Use the scaled data
                self._model.fit(self._xTrainScaled, self._yTrain)
                # self._model.fit(self._xTrain, self._yTrain)
            except ConvergenceWarning:
                self.log.error(f"Failed to converge: {self._selections}")

        self._scores = cross_val_score(self._model, self._x.values.tolist(), self._y)
        #self._scores = cross_val_score(self._model, self._x, self._y)

        self._y_scores = self._model.predict_proba(self._xTestScaled)
        self._fpr, self._tpr, self._threshold = roc_curve(self._yTest, self._y_scores[:, 1])
        self._auc = auc(self._fpr, self._tpr)

        if score:
            self.log.debug(f"MLP cross validation scores: {self._scores}")

            print("Multi-layer perceptron")
            print(self._model.predict(self._xTest))
            print("Training Score: {:.3f}".format(self._model.score(self._xTrain, self._yTrain)))
            print("Testing Score: %s" % self._model.score(self._xTest, self._yTest))

    # Overload the function for the MLP to reference the scaled data
    def accuracy(self) -> float:
        self._model.predict(self._xTestScaled)
        return self._model.score(self._xTestScaled, self._yTest)

    def classify(self):
        self.log.debug(f"Classify by MLP")
        #super().classify(constants.REASON_MLP)
        self._prepareData()

        predictions = self._model.predict(self._blobsInView)
         # Mark up the current view
        i = 0
        for blobName, blobAttributes in self._blobs.items():
            if blobAttributes[constants.NAME_REASON] != constants.REASON_AT_EDGE:
                blobAttributes[constants.NAME_TYPE] = predictions[i]
                blobAttributes[constants.NAME_REASON] = constants.REASON_RANDOM_FOREST
            i = i + 1
        return

class LDA(Classifier):
    name = ClassificationTechniques.LDA.name

    def __init__(self):
        super().__init__()

    def createModel(self, score: Score):
        """
        Create an LDA model
        :param score: True indicates a text output of the score
        """
        self._model = LinearDiscriminantAnalysis()
        self._model.fit(self._xTrain, self._yTrain)

        self._scores = cross_val_score(self._model, self._x, self._y)

        self._y_scores = self._model.predict_proba(self._xTest)
        self._fpr, self._tpr, self._threshold = roc_curve(self._yTest, self._y_scores[:, 1])
        self._auc = auc(self._fpr, self._tpr)

        if score:
            self.log.debug(f"LDA cross validation scores: {self._scores}")

            print("Linear Discriminate Analysis")
            print(self._model.predict(self._xTest))
            print("Training Score: {:.3f}".format(self._model.score(self._xTrain, self._yTrain)))
            print("Testing Score: %s" % self._model.score(self._xTest, self._yTest))


    def classify(self):
        # TODO: Raise an exception
        if self._model is None:
            return

        self.log.info("Classify by LDA")
        self._prepareData()

        # Make the predictions using the model trained
        predictions = self._model.predict(self._blobsInView)

        # Put the predictions into the blobs and mark the reason as LDA
        i = 0
        for blobName, blobAttributes in self._blobs.items():
            if blobAttributes[constants.NAME_REASON] != constants.REASON_AT_EDGE:
                blobAttributes[constants.NAME_TYPE] = predictions[i]
                blobAttributes[constants.NAME_REASON] = constants.REASON_LDA
            i = i + 1
        return

class SuppportVectorMachineClassifier(Classifier):
    name = ClassificationTechniques.SVM.name

    def __init__(self):
        super().__init__()

    def createModel(self, score: bool):
        """
        Create the model for SVM
        :param score: A boolean indicating if scores should be printed.
        :return:
        """

        # Confirm the file exists
        # if not os.path.isfile(filename):
        #     raise FileNotFoundError

        # # Load from the csv file and get the columns we care about
        # self._df = pd.read_csv(filename, usecols=["ratio", "shape", "distance", "type"])
        # # Extract the type -- there should be only two, desired and undesired
        # y = self._df.type
        # # Drop the type column
        # self._df.drop("type", axis='columns', inplace=True)
        # # Split up the data
        # X_train, X_test, y_train, y_test = train_test_split(self._df,y,train_size=0.4)

        # The default of 1000 iterations fails to converge
        # dual==True fails to converge
        self._model = LinearSVC(dual=False, max_iter=10000)
        self._model.fit(self._xTrain, self._yTrain)

        self._scores = cross_val_score(self._model, self._x, self._y)

        # Since linearSVC doesn't support the probabilities needed, use this approach
        clf = CalibratedClassifierCV(self._model)
        clf.fit(self._xTrain, self._yTrain)
        y_proba = clf.predict_proba(self._xTest)

        self._y_scores = clf.predict_proba(self._xTest)
        self._fpr, self._tpr, self._threshold = roc_curve(self._yTest, self._y_scores[:, 1])
        self._auc = auc(self._fpr, self._tpr)

        # Debug
        if score:
            self.log.debug(f"SVM cross validation scores: {self._scores}")

            print("Support Vector Machine prediction")
            print(self._model.predict(self._xTest))
            print("Training Score: {:.3f}".format(self._model.score(self._xTrain, self._yTrain)))
            print("Testing Score: %s" % self._model.score(self._xTest, self._yTest))

        return

    def classify(self):
        """
        Classify the blobs in the image
        """
        # TODO: Raise an exception
        if self._model is None:
            return

        self.log.info("Classify by support vector machine")
        self._prepareData()

        # Make the predictions using the model trained
        predictions = self._model.predict(self._blobsInView)

        # Put the predictions into the blobs and mark the reason as SVM
        i = 0
        for blobName, blobAttributes in self._blobs.items():
            if blobAttributes[constants.NAME_REASON] != constants.REASON_AT_EDGE:
                blobAttributes[constants.NAME_TYPE] = predictions[i]
                blobAttributes[constants.NAME_REASON] = constants.REASON_SVM
            i = i + 1
        return


class LogisticRegressionClassifier(Classifier):
    name = ClassificationTechniques.LOGISTIC.name

    def __init__(self):
        super().__init__()

    def createModel(self, score: bool):
        """
        Initialize the logistic regression subsystem.
        :param score:
        :param filename: A csv format file name with type, ratio, shape, and area columns
        :return:
        """
        # Confirm the file exists
        # if not os.path.isfile(filename):
        #     raise FileNotFoundError

        # # Load from the csv file and get the columns we care about
        # self._df = pd.read_csv(filename, usecols=["ratio", "shape", "distance", "type"])
        # # Extract the type -- there should be only two, desired and undesired
        # y = self._df.type
        # # Drop the type column
        # self._df.drop("type", axis='columns', inplace=True)
        # # Split up the data
        # X_train, X_test, y_train, y_test = train_test_split(self._df,y,train_size=0.4)

        self.log.debug("Creating LR Model")
        self._model = LogisticRegression(C=100, max_iter=300)
        self._model.fit(self._xTrain, self._yTrain)

        self._scores = cross_val_score(self._model, self._x, self._y)

        self._y_scores = self._model.predict_proba(self._xTest)
        self._fpr, self._tpr, self._threshold = roc_curve(self._yTest, self._y_scores[:, 1])
        self._auc = auc(self._fpr, self._tpr)

        # Debug
        if score:
            self.log.debug(f"LR Cross validation scores: {self._scores}")
            print("Logistic regression prediction")
            print(self._model.predict(self._xTest))
            print("Training Score: {:.3f}".format(self._model.score(self._xTrain, self._yTrain)))
            print("Testing Score: %s" % self._model.score(self._xTest,self._yTest))

        return

    def scatterPlotDataset(self):
        """
        Display a scatterplot Matrix of the dataset
        """
        vegetationDataframe = pd.DataFrame(self._xTrain, columns=["ratio", "shape", "distance"])
        pd.plotting.scatter_matrix(vegetationDataframe,
                                   c = self._yTrain,
                                   figsize=(15,15),
                                   marker='o',
                                   hist_kwds={'bins': 20},
                                   s = 60,
                                   alpha=.8)
        plt.show()

    def classify(self):
        """
        Use logistic regression to classify objects in the image as desired on undesired
        :return:
        """
        # TODO: Raise an exception
        if self._model is None:
            return

        self.log.info("Classify by logistic regression")
        self._prepareData()

        # Make the predictions using the model trained
        predictions = self._model.predict(self._blobsInView)

        # Put the predictions into the blobs and mark the reason as logistic regression
        i = 0
        for blobName, blobAttributes in self._blobs.items():
            if blobAttributes[constants.NAME_REASON] != constants.REASON_AT_EDGE:
                blobAttributes[constants.NAME_TYPE] = predictions[i]
                blobAttributes[constants.NAME_REASON] = constants.REASON_LOGISTIC_REGRESSION
            i = i + 1
        return

class KNNClassifier(Classifier):
    name = ClassificationTechniques.KNN.name

    def __init__(self):
        super().__init__()

    def createModel(self, score: bool):

        self._model = KNeighborsClassifier(n_neighbors=5)
        #model = forest.fit(train_fold, train_y.values.ravel())
        self._model.fit(self._xTrain, pd.DataFrame(self._yTrain).values.ravel())

        self._scores = cross_val_score(self._model, self._x, self._y)

        self._y_scores = self._model.predict_proba(self._xTest)
        self._fpr, self._tpr, self._threshold = roc_curve(self._yTest, self._y_scores[:, 1])
        self._auc = auc(self._fpr, self._tpr)

        if score:
            self.log.debug(f"KNN Cross validation scores: {self._scores}")
            yPred = self._model.predict(self._xTest)
            self.log.debug(f"K Neighbors prediction:\n{yPred}")

            self.log.debug("Training Score: {:.3f}".format(self._model.score(self._xTrain, self._yTrain)))
            self.log.debug("Testing Score: {:.2f}\n".format(self._model.score(self._xTest, self._yTest)))


    def classify(self):
        if self._model is None:
            return

        # Make a dataframe out of the current view
        self._prepareData()

        # Predict the types
        predictions = self._model.predict(self._blobsInView)

        # Mark up the current view
        i = 0
        for blobName, blobAttributes in self._blobs.items():
            if blobAttributes[constants.NAME_REASON] != constants.REASON_AT_EDGE:
                blobAttributes[constants.NAME_TYPE] = predictions[i]
                blobAttributes[constants.NAME_REASON] = constants.REASON_KNN
            i = i + 1

        return


class DecisionTree(Classifier):
    name = ClassificationTechniques.DECISIONTREE.name

    def __init__(self):
        super().__init__()


    def visualize(self):
        print("Feature Importance")
        print(self._classifier.feature_importances_)
        #return
        nFeatures = self._df.shape[1]
        plt.barh(np.arange(nFeatures),
                 self._classifier.feature_importances_,
                 align="center")
        #plt.yticks(np.arange(nFeatures))
        plt.show()

    def createModel(self, score: bool):
        self._model = DecisionTreeClassifier(random_state=0)
        self._model.fit(self._xTrain, self._yTrain)

        self._scores = cross_val_score(self._model, self._x, self._y)

        self._y_scores = self._model.predict_proba(self._xTest)
        self._fpr, self._tpr, self._threshold = roc_curve(self._yTest, self._y_scores[:, 1])
        self._auc = auc(self._fpr, self._tpr)

        if score:
            self.log.debug(f"Decision Tree cross validation scores: {self._scores}")

            print("Training Score: {:.3f}".format(self._model.score(self._xTrain, self._yTrain)))
            print("Testing Score: {:.2f}\n".format(self._model.score(self._xTest, self._yTest)))
        return

    def classify(self):

        self._prepareData()

        predictions = self._model.predict(self._blobsInView)
                # Mark up the current view
        i = 0
        for blobName, blobAttributes in self._blobs.items():
            if blobAttributes[constants.NAME_REASON] != constants.REASON_AT_EDGE:
                blobAttributes[constants.NAME_TYPE] = predictions[i]
                blobAttributes[constants.NAME_REASON] = constants.REASON_DECISION_TREE
            i = i + 1
        return

class RandomForest(Classifier):
    name = ClassificationTechniques.RANDOMFOREST.name

    def __init__(self):
        super().__init__()

    def createModel(self, score: bool):
        #self._classifier = RandomForestClassifier(n_estimators=1000, max_features=1, random_state=2, n_jobs=-1)
        self._model = RandomForestClassifier(n_estimators=1000, max_features=1, random_state=2, n_jobs=-1)
        self._model.fit(self._xTrain, self._yTrain)

        self._scores = cross_val_score(self._model, self._x, self._y, n_jobs=-1)

        self._y_scores = self._model.predict_proba(self._xTest)
        self._fpr, self._tpr, self._threshold = roc_curve(self._yTest, self._y_scores[:, 1])
        self._auc = auc(self._fpr, self._tpr)

        if score:
            self.log.debug(f"Random Forest cross validation scores: {self._scores}")

            print("Training Score: {:.3f}".format(self._model.score(self._xTrain, self._yTrain)))
            print("Testing Score: {:.3f}".format(self._model.score(self._xTest, self._yTest)))
        return

    def classify(self):
        self._prepareData()

        predictions = self._model.predict(self._blobsInView)
                # Mark up the current view
        i = 0
        for blobName, blobAttributes in self._blobs.items():
            if blobAttributes[constants.NAME_REASON] != constants.REASON_AT_EDGE:
                blobAttributes[constants.NAME_TYPE] = predictions[i]
                blobAttributes[constants.NAME_REASON] = constants.REASON_RANDOM_FOREST
            i = i + 1
        return

class GradientBoosting(Classifier):
    name = ClassificationTechniques.GRADIENT.name

    def __init__(self):
        super().__init__()

    def createModel(self, score: bool):
        self._model = GradientBoostingClassifier(random_state=0, max_depth=4)
        self._model.fit(self._xTrain, self._yTrain)

        self._scores = cross_val_score(self._model, self._x, self._y)

        self._y_scores = self._model.predict_proba(self._xTest)
        self._fpr, self._tpr, self._threshold = roc_curve(self._yTest, self._y_scores[:, 1])
        self._auc = auc(self._fpr, self._tpr)

        if score:
            self.log.debug(f"Gradient Boosting cross validation scores: {self._scores}")

            print("Training Score: {:.3f}".format(self._model.score(self._xTrain, self._yTrain)))
            print("Testing Score: {:.3f}".format(self._model.score(self._xTest, self._yTest)))


    def classify(self):
        super().classify(constants.REASON_GRADIENT)
        return

class SVM(Classifier):

    def createModel(self, score: bool):
        raise NotImplementedError


    def visualize(self):
        raise NotImplementedError
        return

    def classify(self):
        raise NotImplementedError
        return



def classifierFactory(technique: str) -> Classifier:

    classifiers = {
        RandomForest.name.upper(): RandomForest,
        GradientBoosting.name.upper(): GradientBoosting,
        LDA.name.upper(): LDA,
        MLP.name.upper(): MLP,
        SuppportVectorMachineClassifier.name.upper(): SuppportVectorMachineClassifier,
        LogisticRegressionClassifier.name.upper(): LogisticRegressionClassifier,
        KNNClassifier.name.upper(): KNNClassifier,
        DecisionTree.name.upper(): DecisionTree
    }

    return classifiers[technique]()
