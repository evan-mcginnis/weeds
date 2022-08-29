#!/bin/sh
#
# GPS
#

if [ cmp -s ~/gpsd /etc/default/gpsd ]; then
  echo "Setting GPS configuration"
  cp ~/gpsd /etc/default/gpsd
else
  echo "GPS already configured."
fi

exit $?
