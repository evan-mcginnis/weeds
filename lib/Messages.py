#
# M E S S A G E S
#

#import xml.etree.ElementTree as ET
import json
import time
#import logging
from abc import ABC, abstractmethod
import numpy as np

import constants
import logging

class MUCMessage(ABC):
    def __init__(self, **kwargs):
        self._asString = ""
        self._data = {}
        self._body = ""
        self._timestamp = 0
        self._json = ""
        self._log = logging.getLogger(__name__)

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

    # @abstractmethod
    # def parse(self) -> bool:
    #     """
    #     Parse the message
    #     :return: False on invalid body, true otherwise
    #     """
    #     #self._root = ET.fromstring(self._body)
    #     return self._root is not None

    def formMessage(self) -> str:
        """
        Create an JSON Message
        :return: String of the message
        """
        if self._timestamp == 0:
            self.timestamp = time.time() * 1000
        self._json = json.dumps(self._data)
        return self._json

class SystemMessage(MUCMessage):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if constants.JSON_NAME in self._data:
            self._name = self._data[constants.JSON_NAME]
        else:
            self._name = ""
        if constants.JSON_ACTION in self._data:
            self._action = self._data[constants.JSON_ACTION]
        else:
            self._action = ""
        if constants.JSON_OPERATION in self._data:
            self._operation = self._data[constants.JSON_OPERATION]
        else:
            self._operation = ""
        if constants.JSON_DIAG_RSLT in self._data:
            self._diagnostics = self._data[constants.JSON_DIAG_RSLT]
        else:
            self._diagnostics = ""
        if constants.JSON_STATUS_CAMERA in self._data:
            self._status_camera = self._data[constants.JSON_STATUS_CAMERA]
        else:
            self._status_camera = ""
        if constants.JSON_STATUS_GPS in self._data:
            self._status_gps = self._data[constants.JSON_STATUS_GPS]
        else:
            self._status_gps = ""
        if constants.JSON_STATUS_SYSTEM in self._data:
            self._status_system = self._data[constants.JSON_STATUS_SYSTEM]
        else:
            self._status_system = ""
        if constants.JSON_STATUS_INTEL in self._data:
            self._status_intel = self._data[constants.JSON_STATUS_INTEL]
        else:
            self._status_intel = ""
        if constants.JSON_STATUS_DAQ in self._data:
            self._status_daq = self._data[constants.JSON_STATUS_DAQ]
        else:
            self._status_camera = ""
        if constants.JSON_PARAM_GSD in self._data:
            self._param_gsd = self._data[constants.JSON_PARAM_GSD]
        else:
            self._param_gsd = ""
        if constants.JSON_POSITION in self._data:
            self._position = self._data[constants.JSON_POSITION]
        else:
            self._position = ""

    @property
    def position(self) -> str:
        return self._position

    @position.setter
    def position(self, thePosition):
        self._position = thePosition
        self._data[constants.JSON_POSITION] = thePosition

    @property
    def gsdCamera(self) -> int:
        return int(self._param_gsd)

    @gsdCamera.setter
    def gsdCamera(self, theGSD: str):
        self._param_gsd = theGSD
        self._data[constants.JSON_PARAM_GSD] = theGSD

    @property
    def statusCamera(self) -> str:
        return self._status_camera

    @statusCamera.setter
    def statusCamera(self, theStatus):
        self._status_camera = theStatus
        self._data[constants.JSON_STATUS_CAMERA] = theStatus

    @property
    def statusSystem(self) -> str:
        return self._status_system

    @statusSystem.setter
    def statusSystem(self, theStatus):
        self._status_camera = theStatus
        self._data[constants.JSON_STATUS_SYSTEM] = theStatus

    @property
    def statusGPS(self) -> str:
        return self._status_gps

    @statusGPS.setter
    def statusGPS(self, theStatus):
        self._status_gps = theStatus
        self._data[constants.JSON_STATUS_GPS] = theStatus

    @property
    def statusIntel(self) -> str:
        return self._status_intel

    @statusIntel.setter
    def statusIntel(self, theStatus):
        self._status_intel = theStatus
        self._data[constants.JSON_STATUS_INTEL] = theStatus
    @property
    def statusDAQ(self) -> str:
        return self._status_daq

    @statusDAQ.setter
    def statusDAQ(self, theStatus):
        self._status_daq = theStatus
        self._data[constants.JSON_STATUS_DAQ] = theStatus
    @property
    def diagnostics(self) -> str:
        return self._diagnostics

    @diagnostics.setter
    def diagnostics(self, theDiagnostics):
        self._diagnostics = theDiagnostics
        self._data[constants.JSON_DIAG_RSLT] = theDiagnostics

    @property
    def operation(self) -> str:
        return self._operation

    @operation.setter
    def operation(self, theOperation: str):
        self._operation = theOperation
        self._data[constants.JSON_OPERATION] = theOperation

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
    def action(self, _action: str):
        self._action = _action
        self._data[constants.JSON_ACTION] = _action

    # def parse(self) -> bool:
    #     return super().parse()

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

        if constants.JSON_TYPE in self._data:
            self._type = self._data[constants.JSON_TYPE]
        else:
            self._type = constants.OdometryMessageType.UNKNOWN.name

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

        if constants.JSON_PULSES in self._data:
            self._pulses = self._data[constants.JSON_PULSES]
        else:
            self._pulses = 0

        if constants.JSON_GYRO in self._data:
            self._gyro = self._data[constants.JSON_GYRO]
        else:
            self._gyro = ""

        if constants.JSON_ACCELERATION in self._data:
            self._acceleration = self._data[constants.JSON_ACCELERATION]
        else:
            self._acceleration = ""

        if constants.JSON_GYRO in self._data:
            self._gyro = self._data[constants.JSON_GYRO]
        else:
            self._gyro = ""

        if constants.JSON_DEPTH in self._data:
            self._depth = self._data[constants.JSON_DEPTH]
        else:
            self._depth = 0.0

    @property
    def type(self) -> str:
        """
        The type of message (distance or position)
        :return:
        """
        return self._type

    @type.setter
    def type(self, theType: constants.OdometryMessageType):
        """
        Set the type of the message
        :param theType:
        """
        self._type = theType.name
        self._data[constants.JSON_TYPE] = theType.name

    @property
    def depth(self) -> float:
        """
        The average depth AGL of the camera
        :return: depth as float
        """
        return self._depth

    @depth.setter
    def depth(self, theDepth: float):
        """
        Set the average depth AGL oc the camera
        :param theDepth:
        """
        self._depth = theDepth
        self._data[constants.JSON_DEPTH] = self._depth

    @property
    def gyro(self) -> str:
        return self._gyro

    @gyro.setter
    def gyro(self, values: np.ndarray):
        self._gyro = "({},{},{})".format(values[0], values[1], values[2])
        self._data[constants.JSON_GYRO] = self._gyro

    @property
    def acceleration(self) -> str:
        return self._acceleration

    @acceleration.setter
    def acceleration(self, values: np.ndarray):
        self._acceleration = "({},{},{})".format(values[0], values[1], values[2])
        self._data[constants.JSON_ACCELERATION] = self._acceleration

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
    def pulses(self) -> int:
        return self._pulses

    @pulses.setter
    def pulses(self, pulses: int):
        self._pulses = pulses
        self._data[constants.JSON_PULSES] = pulses

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

    # @property
    # def root(self):
    #     return self._root

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



#
# The treatment message is issued to turn on emitters and to send out the raw plan
#
class TreatmentMessage(MUCMessage):
    def __init__(self, **kwargs):
        """
        A message to the treatment room
        :param body: Message body.
        """
        super().__init__(**kwargs)
        self._name = ""

        if constants.JSON_SOURCE in self._data:
            self._source = self._data[constants.JSON_SOURCE]
        else:
            self._source = ""

        if constants.JSON_PLAN in self._data:
            try:
                self._plan = constants.Treatment[self._data[constants.JSON_PLAN]]
            except KeyError:
                self._log.error("Unable to find mapping for {}".format(self._data[constants.JSON_PLAN]))
        else:
            self._plan = constants.Treatment.UNKNOWN

        if constants.JSON_URL in self._data:
            self._url = self._data[constants.JSON_URL]
        else:
            self._url = ""

        # Left or right position of the emitter
        if constants.JSON_POSITION in self._data:
            self._position = self._data[constants.JSON_POSITION]
        else:
            self._position = constants.EMITTER_NOT_SET

        if constants.JSON_EMITTER_NUMBER in self._data:
            self._emitter_number = self._data[constants.JSON_EMITTER_NUMBER]
        else:
            self._emitter_number = constants.EMITTER_NOT_SET

        # The odometry pulse on which the emitter is activated
        if constants.JSON_PULSE_START in self._data:
            self._pulse_start = self._data[constants.JSON_PULSE_START]
        else:
            self._pulse_start = constants.EMITTER_NOT_SET

        # The odometry pulse on which the emitter is stopped
        if constants.JSON_PULSE_STOP in self._data:
            self._pulse_stop = self._data[constants.JSON_PULSE_STOP]
        else:
            self._pulse_stop = constants.EMITTER_NOT_SET


        # The position of the emitter within the tier
        if constants.JSON_EMITTER_POS in self._data:
            self._emitter_position = self._data[constants.JSON_EMITTER_POS]
        else:
            self._emitter_position = constants.EMITTER_NOT_SET

        # The emitter tier
        if constants.JSON_EMITTER_TIER in self._data:
            self._emitter_tier = self._data[constants.JSON_EMITTER_TIER]
        else:
            self._emitter_tier = constants.EMITTER_NOT_SET

        # The duration of the emitter pulse in milliseconds
        if constants.JSON_EMITTER_DURATION in self._data:
            self._emitter_duration = self._data[constants.JSON_EMITTER_DURATION]
        else:
            self._emitter_duration = constants.EMITTER_NOT_SET

    @property
    def source(self) -> str:
        """
        The source of the capture
        :return: Name from Capture enum
        """
        return self._source

    @source.setter
    def source(self, theSource: constants.Capture):
        """
        Set the name of the source of capture
        :param theSource:
        """
        self._source = theSource.name
        self._data[constants.JSON_SOURCE] = theSource.name

    @property
    def position(self) -> str:
        """
        The position of the emitter to activate
        :return:
        """
        return self._position

    @position.setter
    def position(self, thePosition: str):
        self._position = thePosition
        self._data[constants.JSON_POSITION] = thePosition

    @property
    def url(self) -> str:
        """
        The URL of the image of the plan
        :return:
        A string of the image representing the plan
        """
        return self._url

    @url.setter
    def url(self, theUrl : str):
        """
        Set the URL of the image
        :param theUrl: The URM to use
        """
        self._url = theUrl
        self._data[constants.JSON_URL] = theUrl

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
    def pulse_start(self) -> int:
        """
        The odometry pulse value on which the emitter is to be activated
        :return:
        """
        return int(self._pulse_start)

    @pulse_start.setter
    def pulse_start(self, thePulse: int):
        """
        Set the odometry pulse on which the emitter is to be activated. Use 0 to signal immediate activation
        :param theDuration: The pulse value to use or 0 (zero)
        """
        self._pulse_start = thePulse
        self._data[constants.JSON_PULSE_START] = thePulse
    @property
    def pulse_stop(self) -> int:
        """
        The odometry pulse value on which the emitter is to be activated
        :return:
        """
        return int(self._pulse_stop)

    @pulse_stop.setter
    def pulse_stop(self, thePulse: int):
        """
        Set the odometry pulse on which the emitter is to be activated. Use 0 to signal immediate activation
        :param thePulse: The pulse value to use or 0 (zero)
        """
        self._pulse_stop = thePulse
        self._data[constants.JSON_PULSE_STOP] = thePulse

    @property
    def duration(self) -> int:
        """
        The duration of the pulse in seconds. Only used if the pulse to terminate the treatment is not specified.
        :return:
        """

        value = int(self._emitter_duration)
        return value

    @duration.setter
    def duration(self, theDuration: int):
        self._emitter_duration = theDuration
        self._data[constants.JSON_EMITTER_DURATION] =  theDuration

    @property
    def tier(self) -> int:
        """
        The tier (row) of the emitter
        :return:
        """
        return int(self._emitter_tier)

    @tier.setter
    def tier(self, theTier: int):
        """
        Specify the tier (row) of the emitter
        :param theTier:
        """
        self._emitter_tier = theTier
        self._data[constants.JSON_EMITTER_TIER] = theTier

    @property
    def number(self) -> int:
        """
        The number within the tier of the emitter
        :return:
        """
        return int(self._emitter_number)

    @number.setter
    def number(self, theNumber: int):
        """
        Specify the tier (row) of the emitter
        :param theTier:
        """
        self._emitter_number = theNumber
        self._data[constants.JSON_EMITTER_NUMBER] = theNumber

    @property
    def side(self) -> int:
        """
        The position within the weeder (LEFT or RIGHT)
        :return: constants.Position LEFT or RIGHT
        """
        return int(self._emitter_position)

    @side.setter
    def side(self, thePosition: str):
        """
        Set the position within the tier weeder (LEFT OR RIGHT)
        :param thePosition: use constants.Position for LEFT or RIGHT
        """
        self._emitter_position = constants.Side[thePosition].value
        self._data[constants.JSON_EMITTER_POS] = self._emitter_position

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



