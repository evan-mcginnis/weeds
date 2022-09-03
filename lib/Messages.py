#
# M E S S A G E S
#

#import xml.etree.ElementTree as ET
import json
import logging
from abc import ABC, abstractmethod

import constants


class MUCMessage(ABC):
    def __init__(self, **kwargs):
        self._asString = ""
        self._data = {}
        self._body = ""
        self._timestamp = 0

        # This is the case where a dictionary is passed in with all the data
        # _body contains a string & _data contains a dictionary
        if constants.JSON_DATA in kwargs:
            self._body = json.dumps(kwargs.get(constants.JSON_DATA))
        # This is the case where a message is pulled out of the chatroom
        # _data contains a dictionary
        elif constants.MSG_RAW in kwargs:
            try:
                self._data = json.loads(kwargs.get(constants.MSG_RAW))
            except TypeError as e:
                print("Unable to load: {}".format(kwargs.get(constants.MSG_RAW)))

        # If the init parameters contained the time, set it here.
        if constants.JSON_TIME in self._data:
            self._timestamp = self._data[constants.JSON_TIME]

        return

    @property
    def timestamp(self) -> int:
        return self._timestamp

    @timestamp.setter
    def timestamp(self, stamp: int):
        self._timestamp = stamp
        self._data[constants.JSON_TIME] = stamp

    @property
    def data(self) -> []:
        return self._data

    @data.setter
    def data(self, document: str):
        self._body = document

    @abstractmethod
    def parse(self) -> bool:
        """
        Parse the message
        :return: False on invalid body, true otherwise
        """
        #self._root = ET.fromstring(self._body)
        return self._root is not None

    def formMessage(self) -> str:
        """
        Create an JSON Message
        :return: String of the message
        """
        self._json = json.dumps(self._data)
        return self._json

class SystemMessage(MUCMessage):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if constants.JSON_NAME in self._data:
            self._name = self._data[constants.JSON_NAME]
        if constants.JSON_ACTION in self._data:
            self._action = self._data[constants.JSON_ACTION]

    @property
    def name(self):
        return self._name

    # The name of this run
    @name.setter
    def name(self, target: str):
        self._data[constants.JSON_NAME] = target
        self._name = target

    @property
    def action(self):
        return self._action

    # START, STOP, DIAGNOSE
    @action.setter
    def action(self, _action: constants.Action):
        self._action = _action
        self._data[constants.JSON_ACTION] = _action.name

    def parse(self) -> bool:
        return super().parse()

class DiagnosticMessage(SystemMessage):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._result = True
        self._details = ""

        if constants.JSON_DIAG_RSLT in self._data:
            self._result = (self._data[constants.JSON_DIAG_RSLT] == constants.DIAG_PASS)
        if constants.JSON_DIAG_DETAIL in self._data:
            self._details = self._data[constants.JSON_DIAG_DETAIL]

    @property
    def result(self) -> bool:
        return self._result

    @result.setter
    def result(self, results: bool):
        self._result = results

    @property
    def details(self):
        return self._details

    @details.setter
    def details(self, details: str):
        self._details = details



class OdometryMessage(MUCMessage):
    def __init__(self, **kwargs):
        """
        A message to the odometry room
        :param body: Message body.
        """
        super().__init__(**kwargs)

        # If the init parameters contained the distance, set it here.
        if constants.JSON_DISTANCE in self._data:
            self._distance = self._data[constants.JSON_DISTANCE]
        else:
            self._distance = 0

        if constants.JSON_TOTAL_DISTANCE in self._data:
            self._totalDistance = self._data[constants.JSON_TOTAL_DISTANCE]
        else:
            self._totalDistance = 0.0

        if constants.JSON_SPEED in self._data:
            self._speed = self._data[constants.JSON_SPEED]
        else:
            self._speed = 0.0

        if constants.JSON_SOURCE in self._data:
            self._source = self._data[constants.JSON_SOURCE]
        else:
            self._source = ""

        if constants.JSON_ACTION in self._data:
            self._action = self._data[constants.JSON_ACTION]
        else:
            self._action = ""

        if constants.JSON_LATITUDE in self._data:
            self._latitude = self._data[constants.JSON_LATITUDE]
        else:
            self._latitude = 0.0

        if constants.JSON_LONGITUDE in self._data:
            self._longitude = self._data[constants.JSON_LONGITUDE]
        else:
            self._longitude = 0.0

    @property
    def latitude(self) -> float:
        return self._latitude

    @latitude.setter
    def latitude(self, lat: float):
        self._latitude = lat
        self._data[constants.JSON_LATITUDE] = lat

    @property
    def longitude(self) -> float:
        return self._longitude

    @longitude.setter
    def longitude(self, long: float):
        self._longitude = long
        self._data[constants.JSON_LONGITUDE] = long

    @property
    def speed(self) -> float:
        return self._speed

    @speed.setter
    def speed(self, speed: int):
        self._speed = speed
        self._data[constants.JSON_SPEED] = speed

    @property
    def distance(self):
        return self._distance

    @distance.setter
    def distance(self, distanceTravelled: int):
        self._distance = distanceTravelled
        self._data[constants.JSON_DISTANCE] = distanceTravelled

    @property
    def totalDistance(self) -> float:
        return self._totalDistance

    @totalDistance.setter
    def totalDistance(self, distanceTravelled: float):
        self._totalDistance = distanceTravelled
        self._data[constants.JSON_TOTAL_DISTANCE] = distanceTravelled

    @property
    def source(self) -> str:
        return self._source

    @source.setter
    def source(self, sourceOfReading: str):
        self._source = sourceOfReading
        self._data[constants.JSON_SOURCE] = sourceOfReading

    @property
    def action(self) -> str:
        return self._action

    @action.setter
    def action(self, requestedAction: str):
        self._action = requestedAction
        self._data[constants.JSON_ACTION] = requestedAction

    @property
    def root(self):
        return self._root

    @property
    def body(self):
        return self._body

    @body.setter
    def body(self, document: str):
        self._body = document

    @property
    def message(self) -> str:
        return self._asString

    def formMessage(self) -> str:
        """
        Create an JSON Message
        :return: String of the message
        """
        # XML version
        # self._root = ET.Element(constants.XML_ROOT)
        # distance = ET.SubElement(self._root, constants.XML_DISTANCE)
        # distance.text = str(self._distance)
        #
        # self._asString = ET.tostring(self._root, encoding='us-ascii', method='xml')
        # return self._asString.decode()
        self._json = json.dumps(self._data)
        return self._json

    def parse(self) -> bool:
        super().parse()

class TreatmentMessage(MUCMessage):
    def __init__(self, **kwargs):
        """
        A message to the odometry room
        :param body: Message body.
        """
        super().__init__(**kwargs)
        self._name = ""

        if constants.JSON_PLAN in self._data:
            self._plan = self._data[constants.JSON_PLAN]

    @property
    def plan(self):
        return self._plan

    @plan.setter
    def plan(self, _plan: constants.Treatment):
        self._plan = _plan
        self._data[constants.JSON_PLAN] = _plan.name
        print("Plan is {}".format(_plan.name))

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, _name: str):
        self._name = _name
        self._data[constants.JSON_NAME] = _name

    @property
    def root(self):
        return self._root

    @property
    def body(self):
        return self._body

    @body.setter
    def body(self, document: str):
        self._body = document

    @property
    def message(self) -> str:
        return self._asString

    def formMessage(self) -> str:
        """
        Create an JSON Message
        :return: String of the message
        """

        self._json = json.dumps(self._data)
        return self._json

    def parse(self) -> bool:
        super().parse()

if __name__ == "__main__":

    msg = OdometryMessage()
    msg.distance = 5
    msg.timestamp = 1111
    print("JSON: {}".format(msg.formMessage()))

    msg = OdometryMessage(raw='{"distance": -6}')
    print("JSON: {}".format(msg.formMessage()))



