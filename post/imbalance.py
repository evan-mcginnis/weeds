#
# I M B A L A N C E
#
#
import argparse
import sys

import pandas as pd
import numpy as np

from Classifier import ImbalanceCorrection
from Classifier import classifierFactory
from Classifier import Classifier
from Classifier import LogisticRegressionClassifier, KNNClassifier, DecisionTree, RandomForest, \
    GradientBoosting, SuppportVectorMachineClassifier, LDA, MLP
from Classifier import ClassificationTechniques
from Classifier import Subset
from OptionsFile import OptionsFile
import constants

import logging
import logging.config

class Imbalances:
    def __init__(self):
        self._df = pd.DataFrame(columns=['classification', 'correction', 'minority', 'auc'])
        self._results = []

    def recordResult(self,
                     classificationTechnique: str,
                     correctionTechnique: str,
                     minorityClassRepresentation: float,
                     auc: float):
        self._results.append({'classification': classificationTechnique,
                             'correction': correctionTechnique,
                             'minority': minorityClassRepresentation,
                             'auc': auc})

    def write(self, outputFile: str):
        self._df = pd.DataFrame(self._results)
        self._df.to_csv(outputFile)

imbalanceCorrectionChoices = [i.name.lower() for i in ImbalanceCorrection]
imbalanceCorrectionChoices.append("ALL")

classificationChoices = [i.name.lower() for i in ClassificationTechniques]
classificationChoices.append("ALL")

subsetChoices = [i.name.lower() for i in Subset]

parser = argparse.ArgumentParser("Imbalance analysis")

parser.add_argument('-o', '--output', action="store", required=False, default="imbalance.csv", help="Output CSV")
parser.add_argument('-a', '--algorithm', action="store", required=False, default=ImbalanceCorrection.SMOTE.name, choices=imbalanceCorrectionChoices)
parser.add_argument('-c', '--classifier', action="store", required=False, default=LogisticRegressionClassifier.name, choices=classificationChoices)
parser.add_argument('-s', '--subset', action="store", required=False, default=Subset.TRAIN.name, choices=subsetChoices, help="Subset to apply correction")
parser.add_argument('-ini', '--ini', action="store", required=False, default=constants.PROPERTY_FILENAME, help="Options INI")
parser.add_argument('-mi', '--minority', action="store", required=False, type=float, default=0.2, help="Adjust minority class to represent this percentage")
parser.add_argument("-lg", "--logging", action="store", default="logging.ini", help="Logging configuration file")
parser.add_argument("-df", "--data", action="store", help="Name of the data in CSV for use in logistic regression or KNN")
arguments = parser.parse_args()

logging.config.fileConfig(arguments.logging)
log = logging.getLogger("imbalance")

options = OptionsFile(arguments.ini)
options.load()

selections = [e.strip() for e in options.option(constants.PROPERTY_SECTION_IMAGE_PROCESSING, constants.PROPERTY_FACTORS).split(',')]
log.debug(f"Selected parameters from INI file: {selections}")

if arguments.algorithm == "ALL":
    imbalanceCorrectionChoices = [i.name.lower() for i in ImbalanceCorrection]
else:
    imbalanceCorrectionChoices = [arguments.algorithm.upper()]

if arguments.classifier == "ALL":
    classificationChoices = [i.name for i in ClassificationTechniques]
else:
    classificationChoices = [arguments.classifier.upper()]

if arguments.minority == 0.0:
    minorities = np.linspace(.2, .9, 5)
else:
    minorities = [arguments.minority]


results = Imbalances()

for classificationAlgorithm in classificationChoices:
    for imbalanceAlgorithm in imbalanceCorrectionChoices:
        for minority in minorities:
            log.debug(f"Classify: {classificationAlgorithm} Imbalance: {imbalanceAlgorithm} Minority: {minority}")
            classifier = classifierFactory(classificationAlgorithm)
            classifier.selections = selections
            classifier.correct = True
            classifier.correctSubset = Subset[arguments.subset.upper()]
            classifier.writeDatasetToDisk = True
            classifier.correctionAlgorithm = ImbalanceCorrection[imbalanceAlgorithm.upper()]
            classifier.minority = minority
            log.debug(f"Loaded selections: {classifier.selections}")

            # Only random forest requires the data be stratified
            stratification = isinstance(classifier, RandomForest)

            classifier.load(arguments.data, stratify=stratification)
            classifier.createModel(True)
            results.recordResult(classificationAlgorithm,
                                 imbalanceAlgorithm,
                                 minority,
                                 classifier.auc)

results.write("imbalance.csv")

sys.exit(0)
