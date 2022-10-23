#!/bin/sh
#
# I N S T A L L  R I O
#
# Install the field software on a jetson

# $1 # PULSES of odometer
# $2 name of xmpp server
# $3 size of the wheel

tar xf rio.tar
cd rio
sed --in-place "s/\%PULSES\%/$1/" options.ini
sed --in-place "s/\%SERVER\%/$2/" options.ini
sed --in-place "s/\%SERVER\%/$3/" options.ini

exit $?

