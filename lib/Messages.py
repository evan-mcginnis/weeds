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
            self._data = json.loads(kwargs.get(constants.MSG_RAW))

        # If the init parameters contained the distance, set it here.
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
    def body(self, document: str):
        self._body = document

    @abstractmethod
    def parse(self) -> bool:
        """
        Parse the message
        :return: False on invalid body, true otherwise
        """
        #self._root = ET.fromstring(self._body)
        return self._root is not None

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

    @property
    def distance(self):
        return self._distance

    @distance.setter
    def distance(self, distanceTravelled: int):
        self._distance = distanceTravelled
        self._data[constants.JSON_DISTANCE] = distanceTravelled

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

if __name__ == "__main__":

    msg = OdometryMessage()
    msg.distance = 5
    msg.timestamp = 1111
    print("JSON: {}".format(msg.formMessage()))

    msg = OdometryMessage(raw='{"distance": -6}')
    print("JSON: {}".format(msg.formMessage()))



