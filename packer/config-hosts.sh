#!/bin/sh
#
# Updated hosts file
#

getent hosts controller > /dev/null 2&>1
if [ $? -gt 0 ]; then
  sed -i "2i169.254.212.30 controller" /etc/hosts
fi
getent hosts ubuntu > /dev/null 2&>1
if [ $? -gt 0 ]; then
  sed -i "2i169.254.212.31 ubuntu" /etc/hosts
fi
exit $?