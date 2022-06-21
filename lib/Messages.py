#
# M E S S A G E S
#

import xml.etree.ElementTree as ET
import logging
from abc import ABC, abstractmethod

import constants


class MUCMessage(ABC):
    def __init__(self):
        self._body = None
        self._root = None
        self._asString = ""

        return

    @property
    def body(self) -> str:
        return self._body

    @body.setter
    def body(self, document: str):
        self._body = document

    @abstractmethod
    def parse(self) -> bool:
        """
        Parse the message
        :return: False on invalid body, true otherwise
        """
        self._root = ET.fromstring(self._body)
        return self._root is not None

class OdometryMessage(MUCMessage):
    def __init__(self):
        """
        A message to the odometry room
        :param body: Message body.
        """
        super().__init__()
        self._distance = 0

    @property
    def distance(self):
        return self._distance

    @distance.setter
    def distance(self, distanceTravelled: int):
        self._distance = distanceTravelled

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
        Create an XML Message
        :return: String of the message
        """
        # Create the top level of the document
        self._root = ET.Element(constants.XML_ROOT)
        distance = ET.SubElement(self._root, constants.XML_DISTANCE)
        distance.text = str(self._distance)

        self._asString = ET.tostring(self._root, encoding='us-ascii', method='xml')
        return self._asString.decode()

    def parse(self) -> bool:
        super().parse()

if __name__ == "__main__":

    msg = OdometryMessage()
    msg.distance = 5
    print("XML: {}".format(msg.formMessage()))


