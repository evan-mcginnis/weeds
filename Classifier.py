#
# C L A S S I F I E R
#

import constants
import cv2 as cv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import os.path

class Classifier:

    def __init__(self):

        self._df = pd.DataFrame()

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

    def loadLogisticRequestion(self, filename: str):
        """
        Initialize the logistic regression subsystem.
        :param filename: A csv format file name with type, ratio, shape, and area columns
        :return:
        """
        # Confirm the file exists
        if not os.path.isfile(filename):
            raise FileNotFoundError

        # Load from the csv file and get the columns we care about
        self._df = pd.read_csv(filename, usecols=["ratio", "shape", "distance", "type"])
        # Extract the type -- there should be only two, desired and undesired
        y = self._df.type
        # Drop the type column
        self._df.drop("type", axis='columns', inplace=True)
        # Split up the data
        X_train, X_test, y_train, y_test = train_test_split(self._df,y,train_size=0.4)
        self._model = LogisticRegression()
        self._model.fit(X_train, y_train)

        # Debug
        print("Logistic regression prediction")
        print(self._model.predict(X_test))

        print("Score: %s" % self._model.score(X_test,y_test))

        return

    def classifyByLogisticRegression(self):
        """
        Use logistic regression to classify objects in the image as desired on undesired
        :return:
        """
        # TODO: Raise an exception
        if self._model is None:
            return

        features = []
        # Build up the list of features we will use.
        # Reading some performance comparisons, this is the fastest way to create a dataframe
        for blobName, blobAttributes in self._blobs.items():
            features.append([blobAttributes[constants.NAME_RATIO],
                             blobAttributes[constants.NAME_SHAPE_INDEX],
                             blobAttributes[constants.NAME_DISTANCE]])

        # Construct the dataframe we will use
        blobsInView = pd.DataFrame(features, columns=('ratio', 'shape', 'distance'))

        # Make the predictions using the model trained
        predictions = self._model.predict(blobsInView)

        # Put the predictions into the blobs and mark the reason as logistic regression
        i = 0
        for blobName, blobAttributes in self._blobs.items():
            if blobAttributes[constants.NAME_REASON] != constants.REASON_AT_EDGE:
                blobAttributes[constants.NAME_TYPE] = predictions[i]
                blobAttributes[constants.NAME_REASON] = constants.REASON_LOGISTIC_REGRESSION
            i = i + 1
        return

