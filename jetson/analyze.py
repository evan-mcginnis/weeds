import sys
import os
import logging
import logging.config

import argparse
import pickle

import pandas as pd

import constants
from Classifier import RandomForest
from OptionsFile import OptionsFile

parser = argparse.ArgumentParser("Analyze data file")

parser.add_argument("-df", "--data", action="store", required=True, help="Data file (CSV)")
parser.add_argument("-l", "--logging", action="store", required=False, default="logging.ini", help="Logging configuration")
parser.add_argument('-ini', '--ini', action="store", required=False, default=constants.PROPERTY_FILENAME, help="Options INI")
parser.add_argument('-c', '--correct', action="store_true", required=False, default=False, help="Correct the imbalance")

arguments = parser.parse_args()

if not os.path.isfile(arguments.logging):
    print("Unable to access logging configuration file {}".format(arguments.logging))
    sys.exit(1)

# Initialize logging
logging.config.fileConfig(arguments.logging)
log = logging.getLogger("analyze")

options = OptionsFile(arguments.ini)
options.load()

selections = [e.strip() for e in options.option(constants.PROPERTY_SECTION_IMAGE_PROCESSING, constants.PROPERTY_FACTORS).split(',')]
log.debug(f"Selected parameters from INI file: {selections}")

if os.path.isfile(arguments.data):
    data = pd.read_csv(arguments.data)
else:
    print(f"Unable to access data: {arguments.data}")
    exit(-1)


# Get the counts so we can display the imbalance
type1 = data[data['type'] == 1]
type0 = data[data['type'] == 0]

# Drop the type, as it is not needed anymore
data = data.drop('type', axis=1)
columns = data.columns.tolist()

print("--- Data Attributes ---")
print(f"Entries: {len(data)} Columns: {len(columns)} Type 0: {len(type0)} Type 1: {len(type1)}")
for column in columns:
    zeros = data[data[column] == 0]
    if len(zeros) > 0:
        print(f"Zeros found for: {column}")
        for index, row in zeros.iterrows():
            print(f"{row['name']}")

exit(0)

# The selection technique doesn't really matter -- we just want the class counts
classifier = RandomForest()
classifier.selections = selections
classifier.load(arguments.data, stratify=False)
ratio = classifier.imbalanceRatio()

print(f"Imbalance ratio before correction: {ratio}")

classifier.correctImbalance()
ratio = classifier.imbalanceRatio()
print(f"Imbalance ratio after correction: {ratio}")



sys.exit(0)



