
import argparse
import os.path
import sys
import pandas as pd
import pandas.errors

parser = argparse.ArgumentParser("Combine CSVs")

parser.add_argument("-i", "--input", action="store", required=True, nargs="*", help="Input CSV")
parser.add_argument("-o", "--output", action="store", required=True, help="Output CSV")
parser.add_argument("-f", "--force", action="store_true", required=False, default=False, help="Force overwrite of output file")

arguments = parser.parse_args()

# Refuse to overwrite the file
if os.path.isfile(arguments.output) and not arguments.force:
    print(f"Output file exists: {arguments.output}")
    sys.exit(-1)

data = []
#print(f"Input: {arguments.input}")
for file in arguments.input:
    print(f"Reading: {file}")
    # Confirm the input file exists
    if not os.path.isfile(file):
        print(f"Unable to access: {file}")
        sys.exit(-1)
    try:
        df = pd.read_csv(file)
    except pandas.errors.EmptyDataError:
        print(f"File contains no data: {file}")
        sys.exit(-1)
    except pandas.errors.ParserError:
        print(f"Unable to parse: {file}")
        sys.exit(-1)
    data.append(df)

consolidated = pd.concat(data)

if consolidated is not None:
    try:
        consolidated.to_csv(arguments.output)
    except FileNotFoundError:
        print(f"Unable to write combined files to: {arguments.output}")
        sys.exit(-1)

sys.exit(0)
