#
# T A R G E T S
#
# The list of targets to be treated
#

import json
import constants

class Targets:
    def __init__(self):
        """
        The list of items to be treated, along with support for going to/from JSON encoding
        """
        self._targetsToTreat = []

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)
        # return self.asJSON()

    @property
    def targetList(self) -> []:
        return self._targetsToTreat

    def addTarget(self, location: (), distanceToEdge: float):
        """
        Add a target to be treated in the image
        :param location: A tuple of the (x,y) location
        :param distanceToEdge: The distance in mm to the edge of the image
        """
        treatment = {constants.JSON_TARGET_LOCATION: location, constants.JSON_TARGET_DISTANCE: distanceToEdge}
        self._targetsToTreat.append(treatment)

    def asJSON(self) -> str:
        """
        The current target list as a JSON string
        :return: JSON string
        """
        return json.dumps(self._targetsToTreat)

    def fromJSON(self, rawJSON: str):
        """
        Load the treatment targets from a JSON string
        :param rawJSON: The original JSON
        """
        self._targetsToTreat = json.loads(rawJSON)

if __name__ == "__main__":

    classify = Targets()
    classify.addTarget((100, 100), 200.0)
    classify.addTarget((200, 200), 300.0)
    classify.addTarget((300, 300), 300.0)
    print("Target list JSON: {}".format(classify.asJSON()))
