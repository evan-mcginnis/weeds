#
# I M B A L A N C E
#
#
import argparse
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
        self._df = pd.DataFrame(columns=['classification', 'correction', 'minority', 'auc', 'score'])
        self._results = []
        self._rocResults = {}

    def recordResult(self,
                     classificationTechnique: str,
                     correctionTechnique: str,
                     minorityClassRepresentation: float,
                     classifier: Classifier):
        self._results.append({'classification': classificationTechnique,
                              'correction': correctionTechnique,
                              'minority': minorityClassRepresentation,
                              'auc': classifier.auc,
                              'score': classifier.scores})
        self._rocResults[classifier.name] = {'FPR': classifier.fpr, "TPR": classifier.tpr}

    def visualizeROC(self):
        plt.title(f'Receiver Operating Characteristic')
        for techniqueName, rates in self._rocResults.items():
            plt.plot(rates["FPR"], rates["TPR"], label=techniqueName)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.title(f'ROC Curve')
        plt.show()

    def write(self, outputFile: str):
        self._df = pd.DataFrame(self._results)
        self._df.to_csv(outputFile)

imbalanceCorrectionChoices = [i.name.lower() for i in ImbalanceCorrection]
imbalanceCorrectionChoices.append("ALL")
imbalanceCorrectionChoices.append("NONE")

classificationChoices = [i.name.lower() for i in ClassificationTechniques]
classificationChoices.append("ALL")

subsetChoices = [i.name.lower() for i in Subset]

parser = argparse.ArgumentParser("Imbalance analysis")

parser.add_argument('-o', '--output', action="store", required=False, default="imbalance.csv", help="Output CSV")
parser.add_argument('-a', '--algorithm', action="store", required=False, default=ImbalanceCorrection.SMOTE.name, choices=imbalanceCorrectionChoices)
parser.add_argument('-c', '--classifier', action="store", required=False, default=LogisticRegressionClassifier.name, choices=classificationChoices)
parser.add_argument('-s', '--subset', action="store", required=False, default=Subset.TRAIN.name, choices=subsetChoices, help="Subset to apply correction")
parser.add_argument('-ini', '--ini', action="store", required=False, default=constants.PROPERTY_FILENAME, help="Options INI")
parser.add_argument('-mi', '--minority', action="store", required=False, type=float, default=0.2, help="Adjust minority class to represent this percentage. 0.0 = Choose several values. -1.0 == Do not correct")
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
    minorities = np.linspace(.1, .9, 5)
else:
    minorities = [arguments.minority]


results = Imbalances()

for classificationAlgorithm in classificationChoices:
    for imbalanceAlgorithm in imbalanceCorrectionChoices:
        for minority in minorities:
            log.debug(f"Classify: {classificationAlgorithm} Imbalance: {imbalanceAlgorithm} Minority: {minority}")
            classifier = classifierFactory(classificationAlgorithm)
            classifier.selections = selections
            classifier.correctSubset = Subset[arguments.subset.upper()]
            classifier.writeDatasetToDisk = True
            if arguments.algorithm.upper() == "NONE":
                classifier.correct = False
            else:
                classifier.correct = True
                classifier.correctionAlgorithm = ImbalanceCorrection[imbalanceAlgorithm.upper()]
            if minority > 0.0:
                classifier.minority = minority
            log.debug(f"Loaded selections: {classifier.selections}")

            # Only random forest requires the data be stratified
            stratification = isinstance(classifier, RandomForest)

            classifier.load(arguments.data, stratify=stratification)
            classifier.createModel(True)
            results.recordResult(classificationAlgorithm,
                                 imbalanceAlgorithm,
                                 minority,
                                 classifier)

results.write("imbalance.csv")
results.visualizeROC()

sys.exit(0)
