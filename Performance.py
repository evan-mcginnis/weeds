#
# P E R F O R M A N C E
#
from datetime import datetime

class Performance:
    def __init__(self, performanceFile: str):
        self.times = {}
        self._file = open(performanceFile,"w")

    def start(self):
        self._start = datetime.now()

    def stop(self):
        self._elapsed = datetime.now() - self._start
        self._elapsed_milliseconds = self._elapsed.total_seconds() * 1000

    def stopAndRecord(self, name : str):
        self.stop()
        self._file.write("%s,%s\n" % (name, str(self._elapsed_milliseconds)))

