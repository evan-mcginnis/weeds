#
#
# D I A G N O S T I C S
#

import constants
import json

class Diagnostics:
    def __init__(self, resultsDirectory: str, resultsFile: str):
        """
        Base class for diagnostics
        """
        self._results = constants.Diagnostic.FAIL
        self._resultText = "Diagnostics not yet run"
        self._allResults = []
        self._resultsFile = resultsFile
        self._resultsDirectory = resultsDirectory

    @property
    def results(self) -> constants.Diagnostic:
        return self._results

    @results.setter
    def results(self, theResults: constants.Diagnostic):
        self._results = theResults

    def reset(self):
        self._results = []

    def addText(self, result: str):
        self._allResults.append(result)

    @property
    def resultsFile(self) -> str:
        return self._resultsFile

    def writeHTML(self):
        html = open(self._resultsDirectory + "/" + self._resultsFile, "w")
        html.write(self._resultText)
        html.close()

    def asJSON(self) -> str:
        return json.dumps(self._allResults)

class DiagnosticsJetson(Diagnostics):
    def __init__(self):
        """
        Diagnostics results for left/right jetson systems
        """
        self._camera = constants.Diagnostic.FAIL
        self._cameraText = "Camera not yet tried"
        self._emitters = constants.Diagnostic.FAIL
        self._emittersText = "Emitters not yet diagnosed"

        super().__init__()

    @property
    def camera(self) -> constants.Diagnostic:
        """
        The result of the last camera diagnostics
        :return: a diagnostic result
        """
        return self._camera

    @camera.setter
    def camera(self, result: constants.Diagnostic):
        if result == constants.Diagnostic.FAIL:
            self._results = constants.Diagnostic.FAIL
        self._camera = result
        self._allResults.append({constants.ENTITY_CAMERA: result})

    @property
    def system(self) -> constants.Diagnostic:
        return self._emitters

    @system.setter
    def system(self, result: constants.Diagnostic):
        if result == constants.Diagnostic.FAIL:
            self._results = constants.Diagnostic.FAIL
        self._allResults.append({constants.ENTITY_SYSTEM: result})
        self._emitters = result

class DiagnosticsDAQ(Diagnostics):
    def __init__(self, directory: str, results: str):
        self._camera = constants.Diagnostic.FAIL
        self._cameraText = "Camera not yet tried"
        self._rio = constants.Diagnostic.FAIL
        self._rioText = "RIO not yet tried"
        self._gps = constants.Diagnostic.FAIL
        self._gpsText = "GPS not yet tried"

        super().__init__(directory, results)

    @property
    def camera(self) -> constants.Diagnostic:
        return self._camera

    @camera.setter
    def camera(self, result: constants.Diagnostic):
        if result == constants.Diagnostic.FAIL:
            self._results = constants.Diagnostic.FAIL
        self._allResults.append({constants.ENTITY_CAMERA: result})
        self._camera = result

    @property
    def DAQ(self) -> constants.Diagnostic:
        return self._rio

    @DAQ.setter
    def DAQ(self, result: constants.Diagnostic):
        if result == constants.Diagnostic.FAIL:
            self._results = constants.Diagnostic.FAIL
        self._allResults.append({constants.ENTITY_DAQ: result})
        self._rio = result

    @property
    def gps(self) -> constants.Diagnostic:
        return self._gps

    @gps.setter
    def gps(self, result: constants.Diagnostic):
        if result == constants.Diagnostic.FAIL:
            self._results = constants.Diagnostic.FAIL
        self._allResults.append({constants.ENTITY_GPS: result})
        self._gps = result
