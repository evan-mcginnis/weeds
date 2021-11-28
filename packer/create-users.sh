#!/bin/sh
#
# Create the user if it does not already exist
#

# TODO: Use certificates instead of password
# TODO: move these to variables file
username="weeds"
password="greydog"
shell="/bin/bash"
getent passwd $username > /dev/null 
# TODO: Determine why this line will not work on ubuntu.
# I'm relatively sure there is something I am misunderstanding about shell redirection and
# exit status
#getent passwd $username > /dev/null 2&>1

if [ $? -gt 0 ] ; then
  pass=$(perl -e 'print crypt($password, "password")' $password)
  useradd -m -p "$pass" -s "$shell" "$username"
else
  echo "User exists. No action taken"
fi

exit $?
