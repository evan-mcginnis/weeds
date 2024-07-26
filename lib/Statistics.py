#
# S T A S T I C S
#

from enum import Enum
import math
import logging

import numpy as np
from WeedExceptions import ProcessingError

class Rate(Enum):
    FPR = 0
    FNR = 1
    TPR = 2
    TNR = 3

class Statistics:
    FN = "FN"
    FP = "FP"
    TN = "TN"
    TP = "TP"

    def __init__(self, **kwargs):
        self._countOfDifferences = 0
        self._fn = 0
        self._fp = 0
        self._tn = 0
        self._tp = 0
        self._p = 0
        self._n = 0
        self._log = logging.getLogger(__name__)

        # These are optional, as we may take the defaults and set them later

        if Statistics.FN in kwargs:
            self._fn = int(kwargs[Statistics.FN])
        if Statistics.FP in kwargs:
            self._fp = int(kwargs[Statistics.FP])
        if Statistics.TN in kwargs:
            self._tn = int(kwargs[Statistics.TN])
        if Statistics.TP in kwargs:
            self._TP = int(kwargs[Statistics.TP])

    @property
    def fp(self) -> int:
        """
        False Positives
        :return:
        """
        return self._fp

    @fp.setter
    def fp(self,  theFP: int):
        """
        False Positives
        :param theFP:
        """
        self._fp = theFP

    @property
    def fn(self) -> int:
        """
        False Negatives
        :return:
        """
        return self._fn

    @fn.setter
    def fn(self, theFN: int):
        """
        False Negatives
        :param theFN:
        """
        self._fn = theFN

    @property
    def tp(self) -> int:
        """
        True positives
        :return:
        """
        return self._tp

    @tp.setter
    def tp(self, theTP: int):
        """
        True positives
        :param theTP:
        """
        self._tp = theTP

    @property
    def tn(self) -> int:
        """
        True negatives
        :return:
        """
        return self._tn

    @tn.setter
    def tn(self, theTN: int):
        """
        True negatives
        :param theTN:
        """
        self._tn = theTN

    @property
    def p(self) -> int:
        """
        Total Positives
        :return:
        """
        # Positives
        return self._p

    @p.setter
    def p(self, theP: int):
        """
        Total positives
        :param theP:
        """
        self._p = theP

    @property
    def n(self) -> int:
        """
        Total Negatives
        :return:
        """
        # Negatives
        return self._n

    @n.setter
    def n(self, theN):
        """
        Total Negatives
        :param theN:
        """
        self._n = theN

    def population(self) -> int:
        """
        Population (Negatives + Positives)
        :return:
        """
        return self._p + self._n

    def rate(self, stat: Rate) -> float:
        """
        Determine the specified rate
        :param stat: rate to calculate
        :return:
        """
        # Fornulae taken from
        # https://en.wikipedia.org/wiki/Sensitivity_and_specificity

        rate = 0.0
        self._log.info(f"Calculating rate: {stat}")
        try:
            # False Negative Rate
            if stat == Rate.FNR:
                rate = self._fn / (self._tp + self._fn)

            # False Positive Rate
            elif stat == Rate.FPR:
                rate = self._fp / (self._fp + self._tn)

            # True Positive
            elif stat == Rate.TPR:
                rate = self._tp / (self._tp + self._fn)

            # True Negative
            elif stat == Rate.TNR:
                rate = self._tn / (self._tn + self._fp)

            else:
                raise AttributeError(f"Unsupported rate {stat.name}")
        except ZeroDivisionError:
            self._log.warn(f"Caught division by zero error")

        return rate

    # Formulae for precision, recall, and f1 taken from

    # https://www.geeksforgeeks.org/f1-score-in-machine-learning/

    def precision(self) -> float:
        """
        Precision calculated as tp / (tp + fp)
        :return:
        """
        return self._tp / (self._tp + self._fp)

    def recall(self) -> float:
        """
        Recall calculated as tp / (tp + fn)
        :return:
        """
        return self._tp / (self._tp + self._fn)

    def f1(self) -> float:
        """
        F1 calulated as (2 * precision * recall) / (precision + recall)
        :return:
        """
        #f1Score = (2 * self._tp) / ((2 * self._tp) + self._fp + self._fn)
        f1Score = 2 * ((self.precision() * self.recall()) / (self.precision() + self.recall()))
        return f1Score

    def accuracy(self) -> float:
        totalAccuracy = 0.0
        try:
            totalAccuracy = (self._tp + self._tn) / (self._tp + self._fp + self._fn + self._tn)
        except ZeroDivisionError:
            self._log.error(f"Zero division error computing accuracy")

        return totalAccuracy

    def pp(self) -> int:
        """
        Predicted Positive
        :return:
        """
        return self._tp + self._fp

    def pn(self) -> int:
        """
        Predicted negative
        :return:
        """
        return self._tn + self._fn

    def ppv(self) -> float:
        """
        Positive predictive value
        :return:
        """
        _ppv = 0
        try:
             _ppv = self._tp / self.pp()
        except ZeroDivisionError:
            self._log.error(f"Zero division error computing PPV")

        return _ppv

    def fdr(self) -> float:
        """
        False discovery rate
        :return:
        """
        _fdr = 0.0
        try:
            _fdr = self._fp / self.pp()
        except ZeroDivisionError:
            self._log.error(f"Zero division error computing FDR")
        return _fdr

    def npv(self) -> float:
        _npv = 0.0
        try:
            _npv = self._tn / self.pn()
        except ZeroDivisionError:
            self._log.error(f"Zero division error computing NPV")
        return _npv

    def fm(self) -> float:
        """
        Fowlkes–Mallows index
        :return:
        """
        _fm = math.sqrt(self.ppv() * self.rate(Rate.TPR))
        return _fm

    def fomr(self) -> float:
        """
        False omission rate
        :return:
        """
        rate = 0.0
        try:
            rate = self._fn / self.pn()
        except ZeroDivisionError:
            self._log.error(f"Zero division error computing FOR")
        return rate

    def mcc(self) -> float:
        """
        Matthews correlation coefficient
        :return:
        """
        _mcc = math.sqrt(self.rate(Rate.TPR) * self.rate(Rate.TNR) * self.ppv() * self.npv()) - \
               math.sqrt(self.rate(Rate.FNR) * self.rate(Rate.FPR) * self.fomr() * self.fdr())
        return _mcc

    def prevalence(self) -> float:
        """
        Prevalence
        :return:
        """
        _prevalance = 0.0
        try:
            _prevalance = self._p / (self._p + self._n)
        except ZeroDivisionError:
            self._log.error(f"Zero division error computing prevalence")
        return _prevalance

    def positiveLikelihoodRatio(self) -> float:
        """
        Positive Likelihood ratio
        :return:
        """
        rate = 0.0
        try:
            rate = self.rate(Rate.TPR) / self.rate(Rate.FPR)
        except ZeroDivisionError:
            self._log.error(f"Zero division error computing positive likelihood")
        return rate

    def negativeLikelihoodRatio(self) -> float:
        """
        Negative likelihood ratio
        :return:
        """
        rate = 0.0
        try:
            rate = self.rate(Rate.FNR) / self.rate(Rate.TNR)
        except ZeroDivisionError:
            self._log.error(f"Zero division error computing negative likelihood")
        return rate

    @property
    def differences(self) -> int:
        return self._countOfDifferences

    @differences.setter
    def differences(self, theDifferences: int):
        self._countOfDifferences = theDifferences

    def __str__(self):
        return f"Population: {self.population()} Differences: {self._countOfDifferences} N: {self._n} P: {self._p} FP: {self._fp} FN: {self._fn} TP: {self._tp} TN: {self._tn} F1: {self.f1()}"

if __name__ == "__main__":
    import argparse
    import sys
    import os.path
    import logging.config

    parser = argparse.ArgumentParser("Statistics test")

    parser.add_argument('-l', '--logging', action="store", required=False, default="logging.ini", help="Logging configuration")

    arguments = parser.parse_args()

    if not os.path.isfile(arguments.logging):
        print("Unable to access logging configuration file {}".format(arguments.logging))
        sys.exit(1)

    # Initialize logging
    logging.config.fileConfig(arguments.logging)

    stats = Statistics()

    stats.fn = 5
    stats.fp = 5
    stats.tp = 10
    stats.tn = 10
    stats.p = 5
    stats.n = 5

    print(f"{stats}")
    print(f"FPR: {stats.rate(Rate.FPR)} FNR: {stats.rate(Rate.FNR)} TPR: {stats.rate(Rate.TPR)} TNR: {stats.rate(Rate.TNR)}")

    sys.exit(0)
