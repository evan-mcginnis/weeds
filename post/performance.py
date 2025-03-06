#
# P E R F O R M A N C E
#
# Process the performance data and output a table
#

import argparse
import os
import pandas as pd
import sys

import constants


parser = argparse.ArgumentParser("Process performance results")

parser.add_argument("-i", "--input", action="store", required=True, help="Performance data CSV")
parser.add_argument("-o", "--output", action="store", required=False, help="Output Tex file")

arguments = parser.parse_args()

if arguments.output is not None:
    if os.path.isfile(arguments.output):
        print(f"{arguments.output} exists. Will not overwrite")
        sys.exit(-1)

df = pd.read_csv(arguments.input)

pt = pd.pivot_table(df, values=constants.PERF_TITLE_MICROSECONDS, index=[constants.PERF_TITLE_ACTIVITY], aggfunc='mean')

latex = pt.to_latex(longtable=True)

if arguments.output is not None:
    with open(arguments.output, "w") as f:
        f.write(latex)

print(f"{latex}")

sys.exit(0)






