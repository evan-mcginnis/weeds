from serial import Serial, SerialException
from pyubx2 import UBXReader
import geopy.distance
import argparse
import sys

parser = argparse.ArgumentParser("Determine movement between two samples")

parser.add_argument("-c", '--com', action="store", required=False, default="COM4", help="COM port -- i.e., COM4")

arguments = parser.parse_args()

try:
    stream = Serial(arguments.com, 9600, timeout=3)
except SerialException:
    print(f"Unable to open serial port: {arguments.com}")
    sys.exit(-1)

ubr = UBXReader(stream)
previousLat = 0.0
previousLon = 0.0

# Loop forever. A bit sloppy, as what we should really have here is a keypress detection
while True:
    # Read the data from the UBX chip
    (raw_data, parsed_data) = ubr.read()

    # Only process position messagee
    if parsed_data.identity == "NAV-PVT":
        lat, lon, alt = parsed_data.lat, parsed_data.lon, parsed_data.hMSL
        coordTo = (lat, lon)
        coordFrom = (previousLat, previousLon)
        #print(f"lat = {lat}, lon = {lon}, alt = {alt / 1000} m")

        # Don't bother printing out the distance from the initial dummy position
        if (previousLat != 0.0 and previousLon != 0.0):
            print(f"{geopy.distance.geodesic(coordFrom, coordTo).meters}")

        previousLat = lat
        previousLon = lon

    #print(parsed_data)

