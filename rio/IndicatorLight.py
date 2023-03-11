"""
Example for Adafruit USB tower light w/alarm
don't forge to `pip install pyserial` or `pip3 install pyserial`
"""

import serial
import time

#serialPort = 'COM57'  # Change to the serial/COM port of the tower light
serialPort = '/dev/ttyUSB0'  # on mac/linux, it will be a /dev path
baudRate = 9600

RED_ON = 0x11
RED_OFF = 0x21
RED_BLINK = 0x41

YELLOW_ON= 0x12
YELLOW_OFF = 0x22
YELLOW_BLINK = 0x42

GREEN_ON = 0x14
GREEN_OFF = 0x24
GREEN_BLINK = 0x44

class IndicatorLight:
    def __init__(self, portName: str):
        self._serialPort = portName #'/dev/ttyUSB0'
        self._connected = False

        try:
            self._mSerial = serial.Serial(serialPort, baudRate)
            self._connected = True
        except serial.serialutil.SerialException:
            print("Unable to find light at: {}".format(portName))
            self._connected = False

        # Turn off the lights
        if self._connected:
            sendCommand(self._mSerial, RED_OFF)
            sendCommand(self._mSerial, YELLOW_OFF)
            sendCommand(self._mSerial, GREEN_OFF)

    def diagnostics(self):

        if not self._connected:
            return

        # turn on each LED set in order
        sendCommand(self._mSerial, RED_ON)
        time.sleep(0.5)
        sendCommand(self._mSerial, RED_OFF)

        sendCommand(self._mSerial, YELLOW_ON)
        time.sleep(0.5)
        sendCommand(self._mSerial, YELLOW_OFF)

        sendCommand(self._mSerial, GREEN_ON)
        time.sleep(0.5)
        sendCommand(self._mSerial, GREEN_OFF)

        # Use the built-in blink modes
        sendCommand(self._mSerial, RED_BLINK)
        time.sleep(3)
        sendCommand(self._mSerial, RED_OFF)

        sendCommand(self._mSerial, YELLOW_BLINK)
        time.sleep(3)
        sendCommand(self._mSerial, YELLOW_OFF)

        sendCommand(self._mSerial, GREEN_BLINK)
        time.sleep(3)
        sendCommand(self._mSerial, GREEN_OFF)

    def off(self):
        if self._connected:
            sendCommand(self._mSerial, RED_OFF)
            sendCommand(self._mSerial, YELLOW_OFF)
            sendCommand(self._mSerial, GREEN_OFF)

    @property
    def connected(self):
        return self._connected

    def signalReady(self):
        pass

    def signalOperating(self):
        pass

    def signalWarning(self):
        pass

    def signalError(self):
        pass

def sendCommand(serialport, cmd):
    serialport.write(bytes([cmd]))

if __name__ == '__main__':

    light = IndicatorLight('/dev/ttyUSB0')
    light.diagnostics()
    light.off()



