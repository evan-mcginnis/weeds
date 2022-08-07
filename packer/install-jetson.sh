#!/bin/sh
#
# I N S T A L L  J E T S O N
#
# Install the field software on a jetson

# $1 IP of camera
# $2 name of jetson
# $3 nickname of jetson

tar xf jetson.tar
cd jetson
sed --in-place "s/\%CAMERAIP\%/$1/" options.ini
sed --in-place "s/\%JIDJETSON\%/$2/" options.ini
sed --in-place "s/\%NICKJETSON\%/$3/" options.ini
sed --in-place "s/\%SERVER\%/$4/" options.ini

exit $?

