#!/bin/sh
#
# E J A B B E R D  C O N F I G 
#

# TODO: Determine if a room exists before creating it

# This command should list them
# ejabberdctl muc_online_rooms global

ejabberdctl create_room odometry conference.weeds.com weeds.com
ejabberdctl create_room treatment conference.weeds.com weeds.com
ejabberdctl create_room system conference.weeds.com weeds.com
 
exit $?
