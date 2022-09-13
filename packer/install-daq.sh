#!/bin/sh
#
# I N S T A L L  D A Q
#
# Install the field software on a jetson

# $1 # PULSES of odometer
# $2 name of xmpp server
# $3 position (left, middle, right)

DAQ_HOME=~weeds/rio
SERVICE_DEFINITION_DIR=/etc/systemd/system
ODOMETRY_SERVICE_DEFINITION_FILE=odometry.service
#AWS_SERVICE_DEFINITION_FILE=weeds-uploader.service
PATH_TO_SERVICE_DEFINITION_FILE=$DAQ_HOME/$SERVICE_DEFINITION_FILE
#PATH_TO_AWS_SERVICE_DEFINITION_FILE=$DAQ_HOME/$SERVICE_DEFINITION_FILE
ODOMETRY_SERVICE_NAME=odometry
#AWS_SERVICE_NAME=weeds-uploader

tar xf rio.tar
chown -R weeds rio
chown -R weeds lib
chgrp -R weeds rio
chgrp -R weeds lib
cd rio
sed --in-place "s/\%PULSES\%/$1/" options.ini
sed --in-place "s/\%SERVER\%/$2/" options.ini
sed --in-place "s/\%POSITION\%/$3/" options.ini

#
# S E R V I C E S
#

create_service_if_missing() {
  SERVICE_DEFINITION_FILE=$1
  PATH_TO_SERVICE_DEFINITION_FILE=$DAQ_HOME/$SERVICE_DEFINITION_FILE
  if ! [ -L $SERVICE_DEFINITION_DIR/$SERVICE_DEFINITION_FILE ]; then
    echo "Service definition link does not exist. Creating."
    ln -s $PATH_TO_SERVICE_DEFINITION_FILE $SERVICE_DEFINITION_DIR/$SERVICE_DEFINITION_FILE
  else
    echo "Service definition link in place"
  fi
}

enable_if_not() {
  SERVICE_NAME=$1
  # Determine if the service is enabled
  systemctl is-enabled $SERVICE_NAME
  if [ $? -gt 0 ]; then
    echo "$SERVICE_NAME  is not enabled"
    systemctl enable $SERVICE_NAME
  else
    echo "$SERVICE_NAME already enabled"
  fi
}

create_service_if_missing $ODOMETRY_SERVICE_DEFINITION_FILE
#create_service_if_missing $AWS_SERVICE_DEFINITION_FILE
enable_if_not $ODOMETRY_SERVICE_NAME
#enable_if_not $AWS_SERVICE_NAME

systemctl reload-or-restart $ODOMETRY_SERVICE_NAME
#systemctl reload-or-restart $AWS_SERVICE_NAME

exit $?

