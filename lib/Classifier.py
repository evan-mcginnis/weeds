#
# C L A S S I F I E R
#
import csv
import random

import constants
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import logging
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import LinearSVC

from abc import ABC, abstractmethod

import os.path

class Classifier:
    name = "Base"

    def __init__(self):

        self._df = pd.DataFrame()
        self._rawData = pd.DataFrame()

        self._xTrain = []
        self._yTrain = pd.Series
        self._xTest = []
        self._yTest = []

        self._blobsInView = pd.DataFrame()
        self._selections = []
        self._scores = []
        self._name = "Base"
        self._loaded = False

        self.log = logging.getLogger(__name__)
        # # The ANN for the classifier
        # self._ann = cv.ml.ANN_MLP_create()
        # self._ann.setLayerSizes(np.array([3, 8, 3], np.uint8))
        # # Symetrical sigmoid activation
        # self._ann.setActivationFunction(cv.ml.ANN_MLP_SIGMOID_SYM, 0.6, 1.0)
        # # Backpropogation
        # self._ann.setTrainMethod(cv.ml.ANN_MLP_BACKPROP, 0.1, 0.1)
        # self._ann.setTermCriteria((cv.TERM_CRITERIA_MAX_ITER | cv.TERM_CRITERIA_EPS, 100, 1.0))
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

    # @property
    # def name(self):
    #     return self._name

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

    def classify(self, reason : int):
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
        if not os.path.isfile(filename):
            raise FileNotFoundError

        self.log.info("Load parameter selections")
        # The selection file is just a single line of CSV
        with open(filename) as f:
            reader = csv.reader(f)
            row = next(reader)
        self._selections = row

        return True

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
        self._rawData = self._df

        # Extract the type -- there should be only two, desired and undesired
        y = self._df.type
        self._y = y
        # Drop the type column
        self._df.drop("type", axis='columns', inplace=True)
        self._x = self._df
        # Drop any data that is not part of the factors we want to consider
        # TODO: Put references to height

        # Split up the data
        if stratify:
            X_train, X_test, y_train, y_test = train_test_split(self._df,y,train_size=0.4, stratify=y,random_state=42)
        else:
            X_train, X_test, y_train, y_test = train_test_split(self._df,y,train_size=0.4,random_state=42)
        self._xTrain = X_train
        self._yTrain = y_train
        self._xTest = X_test
        self._yTest = y_test
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
            Xfeatures.append([blobAttributes[constants.NAME_SATURATION],
                             blobAttributes[constants.NAME_I_YIQ],
                             blobAttributes[constants.NAME_BLUE_DIFFERENCE],
                             blobAttributes[constants.NAME_GREYSCALE + constants.DELIMETER + constants.NAME_HOMOGENEITY + constants.DELIMETER + "0"],
                             blobAttributes[constants.NAME_GREYSCALE + constants.DELIMETER + constants.NAME_ENERGY + constants.DELIMETER + "0"],
                             blobAttributes[constants.NAME_GREYSCALE + constants.DELIMETER + constants.NAME_CONTRAST + constants.DELIMETER + "0"],
                             blobAttributes[constants.NAME_GREYSCALE + constants.DELIMETER + constants.NAME_DISSIMILARITY + constants.DELIMETER + "0"],
                             blobAttributes[constants.NAME_GREYSCALE + constants.DELIMETER + constants.NAME_ASM + constants.DELIMETER + "0"]])

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

    def visualize(self):
        return

class SuppportVectorMachineClassifier(Classifier):
    name = "SVM"
    def __init__(self):
        super().__init__()

    def createModel(self, score:bool):
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

        self._model = LinearSVC()
        self._model.fit(self._xTrain, self._yTrain)

        self._scores = cross_val_score(self._model, self._x, self._y)

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
        return


class LogisticRegressionClassifier(Classifier):
    name = "LogisticRegression"

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
    name = "KNN"

    def __init__(self):
        super().__init__()

    def createModel(self, score: bool):

        self._knnModel = KNeighborsClassifier(n_neighbors=5)
        #model = forest.fit(train_fold, train_y.values.ravel())
        self._knnModel.fit(self._xTrain, pd.DataFrame(self._yTrain).values.ravel())

        self._scores = cross_val_score(self._knnModel, self._x, self._y)

        if score:
            self.log.debug(f"KNN Cross validation scores: {self._scores}")
            yPred = self._knnModel.predict(self._xTest)
            print("K Neighbors prediction:\n", yPred)

            print("Training Score: {:.3f}".format(self._knnModel.score(self._xTrain, self._yTrain)))
            print("Testing Score: {:.2f}\n".format(self._knnModel.score(self._xTest, self._yTest)))


    def classify(self):
        if self._knnModel is None:
            return

        # Make a dataframe out of the current view
        self._prepareData()

        # Predict the types
        predictions = self._knnModel.predict(self._blobsInView)

        # Mark up the current view
        i = 0
        for blobName, blobAttributes in self._blobs.items():
            if blobAttributes[constants.NAME_REASON] != constants.REASON_AT_EDGE:
                blobAttributes[constants.NAME_TYPE] = predictions[i]
                blobAttributes[constants.NAME_REASON] = constants.REASON_KNN
            i = i + 1

        return


class DecisionTree(Classifier):
    name = "DecisionTree"

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
        self._classifier = DecisionTreeClassifier(random_state=0)
        self._classifier.fit(self._xTrain, self._yTrain)

        self._scores = cross_val_score(self._classifier, self._x, self._y)

        if score:
            self.log.debug(f"Decision Tree cross validation scores: {self._scores}")

            print("Training Score: {:.3f}".format(self._classifier.score(self._xTrain, self._yTrain)))
            print("Testing Score: {:.2f}\n".format(self._classifier.score(self._xTest, self._yTest)))
        return

    def classify(self):

        self._prepareData()

        predictions = self._classifier.predict(self._blobsInView)
                # Mark up the current view
        i = 0
        for blobName, blobAttributes in self._blobs.items():
            if blobAttributes[constants.NAME_REASON] != constants.REASON_AT_EDGE:
                blobAttributes[constants.NAME_TYPE] = predictions[i]
                blobAttributes[constants.NAME_REASON] = constants.REASON_DECISION_TREE
            i = i + 1
        return

class RandomForest(Classifier):
    name = "RandomForest"

    def __init__(self):
        super().__init__()

    def createModel(self, score: bool):
        #self._classifier = RandomForestClassifier(n_estimators=1000, max_features=1, random_state=2, n_jobs=-1)
        self._classifier = RandomForestClassifier(n_estimators=1000, max_features=1, random_state=2, n_jobs=-1)
        self._classifier.fit(self._xTrain, self._yTrain)

        self._scores = cross_val_score(self._classifier, self._x, self._y, n_jobs=-1)

        if score:
            self.log.debug(f"Random Forest cross validation scores: {self._scores}")

            print("Training Score: {:.3f}".format(self._classifier.score(self._xTrain, self._yTrain)))
            print("Testing Score: {:.3f}".format(self._classifier.score(self._xTest, self._yTest)))
        return

    def visualize(self):
        return

    def classify(self):
        self._prepareData()

        predictions = self._classifier.predict(self._blobsInView)
                # Mark up the current view
        i = 0
        for blobName, blobAttributes in self._blobs.items():
            if blobAttributes[constants.NAME_REASON] != constants.REASON_AT_EDGE:
                blobAttributes[constants.NAME_TYPE] = predictions[i]
                blobAttributes[constants.NAME_REASON] = constants.REASON_RANDOM_FOREST
            i = i + 1
        return

class GradientBoosting(Classifier):
    name = "GradientBoosting"

    def __init__(self):
        super().__init__()
    def createModel(self, score: bool):
        self._classifier = GradientBoostingClassifier(random_state=0, max_depth=4)
        self._classifier.fit(self._xTrain, self._yTrain)

        self._scores = cross_val_score(self._classifier, self._x, self._y)

        if score:
            self.log.debug(f"Gradient Boosting cross validation scores: {self._scores}")

            print("Training Score: {:.3f}".format(self._classifier.score(self._xTrain, self._yTrain)))
            print("Testing Score: {:.3f}".format(self._classifier.score(self._xTest, self._yTest)))

    def visualize(self):
        return

    def classify(self):
        super().classify(constants.REASON_GRADIENT)
        return

class SVM(Classifier):

    def createModel(self, score: bool):
        raise NotImplementedError

        self._classifier = GradientBoostingClassifier(random_state=0, max_depth=4)
        self._classifier.fit(self._xTrain, self._yTrain)
        self._scores = cross_val_score(self._classifier, self._x, self._y)

        if score:
            self.log.debug(f"SVM cross validation scores: {self._scores}")

            print("Training Score: {:.3f}".format(self._classifier.score(self._xTrain, self._yTrain)))
            print("Testing Score: {:.3f}".format(self._classifier.score(self._xTest, self._yTest)))

    def visualize(self):
        raise NotImplementedError
        return

    def classify(self):
        raise NotImplementedError
        return


