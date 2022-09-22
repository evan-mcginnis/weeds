#!/bin/sh
#
# I N S T A L L  J E T S O N
#
# Install the field software on a jetson

# $1 IP of camera
# $2 name of jetson
# $3 nickname of jetson
# $4 XMPP server
# $5 postion (left, middle, right)

JETSON_HOME=~weeds/jetson
HTTP_HOME=~weeds/http
SERVICE_DEFINITION_DIR=/etc/systemd/system
WEEDS_SERVICE_DEFINITION_FILE=weeds.service
#AWS_SERVICE_DEFINITION_FILE=weeds-uploader.service
HTTP_SERVICE_DEFINITION_FILE=weeds-http.service
PATH_TO_SERVICE_DEFINITION_FILE=$JETSON_HOME/$SERVICE_DEFINITION_FILE
#PATH_TO_AWS_SERVICE_DEFINITION_FILE=$JETSON_HOME/$AWS_SERVICE_DEFINITION_FILE
PATH_TO_HTTP_SERVICE_DEFINITION_FILE=$HTTP_HOME/$HTTP_SERVICE_DEFINITION_FILE
WEEDS_SERVICE_NAME=weeds
#AWS_SERVICE_NAME=weeds-uploader
HTTP_SERVICE_NAME=weeds-http

tar xf jetson.tar
chown -R weeds jetson
chown -R weeds lib
chown -R weeds http
#chown -R weeds post
chgrp -R weeds jetson
chgrp -R weeds lib
chgrp -R weeds http
#chgrp -R weeds post
chmod +x jetson/weeds-service

cd jetson
sed --in-place "s/\%CAMERAIP\%/$1/" options.ini
sed --in-place "s/\%JIDJETSON\%/$2/" options.ini
sed --in-place "s/\%NICKJETSON\%/$3/" options.ini
sed --in-place "s/\%SERVER\%/$4/" options.ini
sed --in-place "s/\%NICKJETSON%/$3/" logging.ini
sed --in-place "s/\%POSITION%/$5/" options.ini

#
# S E R V I C E S
#

create_service_if_missing() {
  SERVICE_DEFINITION_FILE=$1
  PATH_TO_SERVICE_DEFINITION_FILE=$2
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

create_service_if_missing $WEEDS_SERVICE_DEFINITION_FILE $PATH_PATH_TO_SERVICE_DEFINITION_FILE
#create_service_if_missing $AWS_SERVICE_DEFINITION_FILE $PATH_TO_AWS_SERVICE_DEFINITION_FILE
create_service_if_missing $HTTP_SERVICE_DEFINITION_FILE $PATH_TO_HTTP_SERVICE_DEFINITION_FILE

systemctl daemon-reload

enable_if_not $WEEDS_SERVICE_NAME
#enable_if_not $AWS_SERVICE_NAME
enable_if_not $HTTP_SERVICE_NAME


systemctl reload-or-restart $WEEDS_SERVICE_NAME
#systemctl reload-or-restart $AWS_SERVICE_NAME
systemctl reload-or-restart $HTTP_SERVICE_NAME

exit $?

