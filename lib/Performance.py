#
# P E R F O R M A N C E
#
from datetime import datetime
import constants as constants

class Performance:
    def __init__(self, performanceFile: str):
        self.times = {}
        self._performanceFile = performanceFile

    def initialize(self) -> bool:
        """
        Initialize the performance data file, truncating it to 0 bytes.
        This will also insert the headers for the data.
        :return: Boolean
        """
        diagnostics = "Performance initialized"
        try:
            file = open(self._performanceFile, "w")
            # clear out any data that is there
            file.truncate(0)
            # Write out the headers for the performance data
            file.write("{},{}\n".format(constants.PERF_TITLE_ACTIVITY, constants.PERF_TITLE_MILLISECONDS))
            file.close()
        except PermissionError:
            diagnostics = "Unable to open: {}\n".format(self._performanceFile)
            return False, diagnostics
        return True, diagnostics

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
        pass
        # The with style of opening the file should make an explicit close unnecessary.
        #self._file.close()