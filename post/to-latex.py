import pandas as pd
import argparse
import os.path

parser = argparse.ArgumentParser("Latex table from CSV")
parser.add_argument("-i", "--input", action="store", required=True, help="Data in CSV format")
parser.add_argument("-lc", "--long", action="store", required=False, default="Long caption", help="Long caption")
parser.add_argument("-sc", "--short", action="store", required=False, default="Short caption", help="Short caption")
parser.add_argument("-l", "--label", action="store", required=False, default="table", help="Table label")
parser.add_argument("-c", "--columnformat", action="store", required=False, help="Column format, i.e., rcl")

arguments = parser.parse_args()

theData = pd.DataFrame()

try:
    if os.path.isfile(arguments.input):
        theData = pd.read_excel(arguments.input)
    else:
        print(f"Unable to access {arguments.input}")
        exit(-1)
except Exception as e:
    print(f"{e}")
    exit(-1)

theData = theData.replace(r"^ +| +$", r"", regex=True)
print(f"{theData.to_latex(longtable=True,index_names=False, index=False, caption=(arguments.long, arguments.short), label=arguments.label,float_format='%.2f', column_format=arguments.columnformat)}")

exit(0)




