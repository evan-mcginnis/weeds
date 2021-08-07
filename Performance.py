#
# P E R F O R M A N C E
#
from datetime import datetime

class Performance:
    def __init__(self, performanceFile: str):
        self.times = {}
        self._performanceFile = performanceFile

    def start(self) -> int:
        """
        Start the performance timer
        :return:
        The current time
        """
        self._start = datetime.now()
        return self._start

    def stop(self) -> int:
        self._elapsed = datetime.now() - self._start
        self._elapsed_milliseconds = self._elapsed.total_seconds() * 1000
        return self._elapsed_milliseconds

    def stopAndRecord(self, name : str):
        self.stop()

        with open(self._performanceFile,"a") as self._file:
            self._file.write("%s,%s\n" % (name, str(self._elapsed_milliseconds)))

    def cleanup(self):
        self._file.close()