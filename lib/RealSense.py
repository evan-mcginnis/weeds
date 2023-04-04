#
# I N T E L  R E A L S E N S E  U T I L I T I E S
#

import pyrealsense2 as rs
import constants
import json

class RealSense:
    def __init__(self):
        self.ctx = rs.context()
        self._devices = []
        self._serial = None
        self._camera = None

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
                self._camera = self.ctx.devices[0]
                found = True

        if not found:
            self._camera = None

        return self._camera

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

    def configure(self, configFile: str):
        # dev = find_device_that_supports_advanced_mode(config)
        jsonDict = json.load(open(configFile))
        jsonString = str(jsonDict).replace("'", '\"')

        h_res = 1280
        v_res = 720
        framerate = 6

        pipeline = rs.pipeline()
        config = rs.config()

        # Configure depth and color streams
        config.enable_stream(rs.stream.depth, h_res, v_res, rs.format.z16, framerate)
        config.enable_stream(rs.stream.color, h_res, v_res, rs.format.rgb8, framerate)

        cfg = pipeline.start(config)
        dev = cfg.get_device()

        advnc_mode = rs.rs400_advanced_mode(self._camera)
        print("Advanced mode is", "enabled" if advnc_mode.is_enabled() else "disabled")
        current = advnc_mode.serialize_json()
        advnc_mode.load_json(jsonString)
        with open(configFile + ".used", "w") as config:
            config.write(current)

        # ser_dev = rs.serializable_device(self._camera)
        # ser_dev.load_json(jsonString)

        print("loaded json from {}".format(configFile))
        # print("Details: {}".format(jsonString))
        # with open(configFile, 'r') as file:
        #     json = file.read().strip()
        #     ser_dev.load_json(json)

if __name__ == "__main__":

    import argparse
    import sys

    parser = argparse.ArgumentParser("Intel RealSense Utility")

    parser.add_argument('-l', '--list', action="store_true", required=False, default=False, help="List devices")
    parser.add_argument('-d', '--details', action="store_true", required=False, default=False, help="Camera details")
    parser.add_argument('-c', '--camera', action="store", required=False, help="Serial number of target device")
    parser.add_argument('-r', '--reset', action="store_true", required=False, default=False, help="Reset camera")
    parser.add_argument('-i', '--input', required=False, help="Load configuration")
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
        if arguments.input:
            sensors.configure(arguments.input)
    sys.exit(rc)


