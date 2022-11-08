#
# M A K E F I L E
#

SPEED?=1

clean:
	rm rio.log*
	rm -rf ../output/*

# To rename the cards on the DAQ, export, then edit, then import

export-daq-config:
	nidaqmxconfig --export daq.cfg

import-daq-config:
	nidaqmxconfig --import daq.cfg --replace

run-all-virtual:
	 python3 controller.py -o -e -r -s $(SPEED) -i virtual.ini

