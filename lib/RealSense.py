#
# I N T E L  R E A L S E N S E  U T I L I T I E S
#

import pyrealsense2 as rs
import constants

class RealSense():
    def __init__(self):
        self.ctx = rs.context()
        self._devices = []
        self._serial = None

    @property
    def devices(self):
        return self._devices

    def device(self, **kwargs) -> rs.device:
        found = False
        camera = None

        try:
            self._serial = str(kwargs[constants.KEYWORD_SERIAL])
        except KeyError as key:
            print("The serial number of the device is not specified by keyword: {}  Using the first device".format(constants.KEYWORD_SERIAL))

        if self._serial is not None:
            for d in self._devices:
                if self._serial == d.get_info(rs.camera_info.serial_number):
                    found = True
                    camera = d
                    break
        else:
            self.query()
            if len(self.ctx.devices) > 0:
                camera = self.ctx.devices[0]
                found = True

        if not found:
            camera = None

        return camera

    def query(self):
        """
        Queries the system for all devices.
        """
        self._devices = self.ctx.query_devices()

    def count(self) -> int:
        """
        The number of devices found in the system
        :return:
        """
        return len(self._devices)

    def list(self):
        """
        Display the list of devices found
        """
        for dev in self._devices:
            print("Device {}".format(dev))

    def reset(self, camera: rs.device):
        """
        Reset the device specified.
        :param camera: The target camera
        """
        camera.hardware_reset()

if __name__ == "__main__":

    import argparse
    import sys

    parser = argparse.ArgumentParser("Intel RealSense Utility")

    parser.add_argument('-l', '--list', action="store_true", required=False, default=False, help="List devices")
    parser.add_argument('-d', '--details', action="store_true", required=False, default=False, help="Camera details")
    parser.add_argument('-c', '--camera', action="store", required=False, help="Serial number of target device")
    parser.add_argument('-r', '--reset', action="store_true", required=False, default=False, help="Reset camera")
    arguments = parser.parse_args()

    sensors = RealSense()
    sensors.query()

    rc = 0
    if arguments.list:
        print("There are {} devices detected".format(sensors.count()))
        sensors.list()

    if arguments.camera is not None:
        camera = sensors.device(serial=arguments.camera)
    else:
        camera = sensors.device()

    if camera is None:
        print("Unable to find camera: {}".format(arguments.camera))
        rc = -1
    else:
        if arguments.reset:
            sensors.reset(camera)
        if arguments.details:
            print("Camera: {}".format(camera))

    sys.exit(rc)


