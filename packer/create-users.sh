#!/bin/sh -X
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
  # Todo -- this probably does not work.  Needs testing
else
  echo "User exists. No action taken"
fi

eval HOME=~$username
grep PYTHONPATH $HOME/.bashrc
if [ $? -gt 0 ] ; then
  echo "Setting python path for user"
  echo "export PYTHONPATH=~/lib" >> $HOME/.bashrc
else
  echo "PYTHONPATH for user detected"
fi

grep OPENBLAS_CORETYPE $HOME/.bashrc
if [ $? -gt 0 ] ; then
  echo "Setting OPENBLAS_CORETYPE for user"
  echo "export OPENBLAS_CORETYPE=ARMv8" >> $HOME/.bashrc
else
  echo "OPENBLAS_CORETYPE detected for user"
fi


exit $?
