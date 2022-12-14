#!/bin/sh
#
# I N S T A L L  C O N T R O L L E R
#
# Install the controller

CONTROLLER_HOME=~weeds/controller
SERVICE_DEFINITION_DIR=/etc/systemd/system
CONTROLLER_SERVICE_DEFINITION_FILE=controller.service
PATH_TO_CONTROLLER_SERVICE_DEFINITION_FILE=$CONTROLLER_HOME/$SERVICE_DEFINITION_FILE
CONTROLLER_SERVICE_NAME=controller

tar xf controller.tar
chown -R weeds controller
chgrp -R weeds controller
chmod +x controller/controller-service

#
# I N I  F I L E
#
# Use the jetson/options.ini or rio/options.ini depending on what platform we are on
# TODO: Restructure things so this is not required
create_ini_file_link() {
  if [ ! -L ~weeds/controller/options.ini ]; then
    if [ -f ~weeds/jetson/options.ini ]; then
      ln -s  ~weeds/jetson/options.ini ~weeds/controller/options.ini
      chown -h weeds:weeds ~weeds/controller/options.ini
    elif [ -f ~weeds/rio/options.ini ]; then
      ln -s  ~weeds/rio/options.ini ~weeds/controller/options.ini
      chown -h weeds:weeds ~weeds/controller/options.ini
    fi
  fi
}

#
# S E R V I C E S
#

create_service_if_missing() {
  SERVICE_DEFINITION_FILE=$1
  PATH_TO_SERVICE_DEFINITION_FILE=$CONTROLLER_HOME/$SERVICE_DEFINITION_FILE
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

create_ini_file_link
create_service_if_missing $CONTROLLER_SERVICE_DEFINITION_FILE
enable_if_not $CONTROLLER_SERVICE_NAME

systemctl reload-or-restart $CONTROLLER_SERVICE_NAME

exit $?

