#
# T I D Y
#
# Tidy the specified dataframe
#
import pandas as pd
from sklearn.feature_selection import VarianceThreshold

class Tidy:
    def __init__(self, target: pd.DataFrame):
        self._original = target
        self._manipulated = None
        self._columnsToDrop = []
        self._uniquenessThreshold = 0.9


    def load(self, filename: str) -> bool:
        """
        Load the dataframe from a CSV.  Not required if dataframe passed in initially
        :param filename: Path name to file
        """
        self._original = pd.read_csv(filename)
        return True

    @property
    def uniqueness(self) -> float:
        return self._uniquenessThreshold

    @uniqueness.setter
    def uniqueness(self, threshold: float):
        if threshold > 1.0 or threshold < 0.0:
            raise ValueError("Threshold must be between 0.0 and 1.0")

        self._uniquenessThreshold = threshold

    @property
    def original(self) -> pd.DataFrame:
        return self._original

    @property
    def manipulated(self) -> pd.DataFrame:
        return self._manipulated

    @property
    def columnsToDrop(self) -> []:
        return self._columnsToDrop

    def analyze(self) -> (bool, str):
        """
        Analyze the original array
        """
        assert self._original is not None
        problemsFound = False
        problemDescription = []
        type1 = self._original[self._original['type'] == 1]
        type0 = self._original[self._original['type'] == 0]
        # Drop the type, as it is not needed anymore
        print(f"Columns: {self._original.columns.tolist()}")
        data = self._original.drop(['type', 'number', 'name', 'agl', 'actual'], errors='ignore', axis=1)
        columns = data.columns.tolist()

        # BEGIN Eliminate columns with low variance
        # This is not quite what we need, as finding the column names is a bit more complicated as this is a numpy arrqy,
        # not a dataframe
        # var_thr = VarianceThreshold(threshold=0.01)  # Removing both constant and quasi-constant
        # var_thr.fit(data)
        # df_transformed = var_thr.transform(data)
        #selectedColumns = df_transformed.columns.toList()
        # END Variance

        #print(f"Entries: {len(data)} Columns: {len(columns)} Type 0: {len(type0)} Type 1: {len(type1)}")

        for column in columns:
            zeros = data[data[column] == 0]
            if len(zeros) > 0:
                problemsFound = True
                problemDescription.append(f"Zeros found in some rows for column {column}")
            #     print(f"Zeros found for: {column}")
            #     self._columnsToDrop.append(column)
            NumberOfUniqueValues = len(data[column].unique())
            uniqueness = NumberOfUniqueValues / len(data[column])
            if uniqueness < self._uniquenessThreshold:
                problemsFound = True
                problemDescription.append(f"Values in column {column} are below threshold @ {uniqueness}")
                self._columnsToDrop.append(column)
            #print(f"Column {column} variance: {data[column].var()}")
        #print(f"Drop: {self._columnsToDrop}")
        return problemsFound, problemDescription

    def clean(self) -> pd.DataFrame:
        """
        Clean up the original array.
        :return: The new array
        """
        return self._manipulated

if __name__ == "__main__":
    import argparse
    import sys
    import os

    parser = argparse.ArgumentParser("Clean up and analyze array")
    parser.add_argument("-i", "--input", required=True, help="Input CSV to load")

    arguments = parser.parse_args()

    if not os.path.isfile(arguments.input):
        print(f"Unable to access: {arguments.input}")
        sys.exit(-1)

    try:
        original = pd.read_csv(arguments.input)
    except FileNotFoundError:
        print("File not found.")
        sys.exit(-1)
    except pd.errors.EmptyDataError:
        print(f"No data found in {arguments.input}")
        sys.exit(-1)
    except pd.errors.ParserError:
        print(f"Parse error for {arguments.input}")
        sys.exit(-1)
    except Exception:
        print(f"Generic exception for {arguments.input}")
        sys.exit(-1)

    m = Tidy(original)
    problemsFound, problemDetails = m.analyze()
    if problemsFound:
        for problem in problemDetails:
            print(problem)
    sys.exit(0)
