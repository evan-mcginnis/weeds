#
# C L A S S I F I E R
#
import random

import constants
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

import os.path

class Classifier:

    def __init__(self):

        self._df = pd.DataFrame()

        self._xTrain = []
        self._yTrain = pd.Series
        self._xTest = []
        self._yTest = []

        self._blobsInView = pd.DataFrame()

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

    @property
    def blobs(self):
        return self._blobs

    @blobs.setter
    def blobs(self, blobSet: {}):
        self._blobs = blobSet

    def classifyWithinCropRow(self):
        return

    #
    # The only real purpose here is to mark the items that are at the edge of the image as such
    #
    def classifyByPosition(self, size : ()):
        """
        Classify items as unknown if they extend off the edge of the image.
        :param size: Unused. Should be refactored out.
        """
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

        predictions = self._classifier.predict(self._blobsInView)
                # Mark up the current view
        i = 0
        for blobName, blobAttributes in self._blobs.items():
            if blobAttributes[constants.NAME_REASON] != constants.REASON_AT_EDGE:
                blobAttributes[constants.NAME_TYPE] = predictions[i]
                blobAttributes[constants.NAME_REASON] = reason
            i = i + 1
        return

    def load(self, filename: str, stratify : bool):
               # Confirm the file exists
        if not os.path.isfile(filename):
            raise FileNotFoundError

        # Load from the csv file and get the columns we care about
        self._df = pd.read_csv(filename,
                               usecols=["ratio",
                                        "shape",
                                        "distance",
                                        constants.NAME_DISTANCE_NORMALIZED,
                                        constants.NAME_HEIGHT,
                                        "type"])
        # Extract the type -- there should be only two, desired and undesired
        y = self._df.type
        self._y = y
        # Drop the type column
        self._df.drop("type", axis='columns', inplace=True)
        # Split up the data
        if stratify:
            X_train, X_test, y_train, y_test = train_test_split(self._df,y,train_size=0.4, stratify=y,random_state=42)
        else:
            X_train, X_test, y_train, y_test = train_test_split(self._df,y,train_size=0.4,random_state=42)
        self._xTrain = X_train
        self._yTrain = y_train
        self._xTest = X_test
        self._yTest = y_test

    def _prepareData(self):

        features = []
        # Build up the list of features we will use.
        # Reading some performance comparisons, this is the fastest way to create a dataframe
        for blobName, blobAttributes in self._blobs.items():
            print("Warning: Faking height data.")
            if blobAttributes[constants.NAME_TYPE] == constants.TYPE_UNDESIRED:
                blobAttributes[constants.NAME_HEIGHT] = random.randint(10, 25)
            else:
                blobAttributes[constants.NAME_HEIGHT] = random.randint(55,75)
            features.append([blobAttributes[constants.NAME_RATIO],
                             blobAttributes[constants.NAME_SHAPE_INDEX],
                             blobAttributes[constants.NAME_DISTANCE],
                             blobAttributes[constants.NAME_DISTANCE_NORMALIZED],
                             blobAttributes[constants.NAME_HEIGHT]])

        # Construct the dataframe we will use
        self._blobsInView = pd.DataFrame(features, columns=('ratio', 'shape', 'distance','normalized_distance', 'height'))

    def visualize(self):
        return


class LogisticRegressionClassifier(Classifier):

    def __init__(self):
        super().__init__()

    def createModel(self, score: bool):
        """
        Initialize the logistic regression subsystem.
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

        self._model = LogisticRegression(C=100)
        self._model.fit(self._xTrain, self._yTrain)


        # Debug
        if score:
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

        self._prepareData()

        # features = []
        # # Build up the list of features we will use.
        # # Reading some performance comparisons, this is the fastest way to create a dataframe
        # for blobName, blobAttributes in self._blobs.items():
        #     features.append([blobAttributes[constants.NAME_RATIO],
        #                      blobAttributes[constants.NAME_SHAPE_INDEX],
        #                      blobAttributes[constants.NAME_DISTANCE]])
        #
        # # Construct the dataframe we will use
        # blobsInView = pd.DataFrame(features, columns=('ratio', 'shape', 'distance'))

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

    def __init__(self):
        super().__init__()

    def createModel(self, score: bool):

        self._knnModel = KNeighborsClassifier(n_neighbors=1)
        self._knnModel.fit(self._xTrain, pd.DataFrame(self._yTrain))

        if score:
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

    def createModel(self, score: bool):
        self._classifier = RandomForestClassifier(n_estimators=1000, max_features=1,random_state=2, n_jobs=-1)
        self._classifier.fit(self._xTrain, self._yTrain)

        if score:
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

    def createModel(self, score: bool):
        self._classifier = GradientBoostingClassifier(random_state=0, max_depth=4)
        self._classifier.fit(self._xTrain, self._yTrain)

        if score:
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

        if score:
            print("Training Score: {:.3f}".format(self._classifier.score(self._xTrain, self._yTrain)))
            print("Testing Score: {:.3f}".format(self._classifier.score(self._xTest, self._yTest)))

    def visualize(self):
        raise NotImplementedError
        return

    def classify(self):
        raise NotImplementedError
        return