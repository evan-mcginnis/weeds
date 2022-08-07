#!/bin/sh
#
# E J A B B E R D  C O N F I G 
#

# TODO: Determine if a room exists before creating it

# This command should list them
# ejabberdctl muc_online_rooms global
# $1 room
# $2 room domain
# $3 server domain
create_room_if_not_present () {
  room=`ejabberdctl muc_online_rooms global | grep $1`
  if [ "$room" = "$1@$2" ]; then
    echo "Room $1 already exists"
  else
   ejabberdctl create_room $1 $2 $3
  fi
}
create_user_if_not_present () {
  user=`ejabberdctl registered-users $2 | grep $1`
  if [ "$user" = "$1" ]; then
    echo "User $1 already exists"
  else
    ejabberdctl register $1 $2 $3
  fi
}

create_room_if_not_present odometry conference.weeds.com weeds.com
create_room_if_not_present treatment conference.weeds.com weeds.com
create_room_if_not_present system conference.weeds.com weeds.com

create_user_if_not_present admin weeds.com greydog
create_user_if_not_present console weeds.com greydog
create_user_if_not_present rio weeds.com greydog
create_user_if_not_present jetson-right weeds.com greydog
create_user_if_not_present jetson-left weeds.com greydog
create_user_if_not_present cloud weeds.com greydog
create_user_if_not_present control weeds.com greydog
create_user_if_not_present debug weeds.com greydog

#ejabberdctl register console weeds.com greydog
#ejabberdctl register rio weeds.com greydog
#ejabberdctl register jetson-right weeds.com greydog
#ejabberdctl register jetson-left weeds.com greydog
#ejabberdctl register cloud weeds.com greydog
#ejabberdctl register control weeds.com greydog
#ejabberdctl register debug weeds.com greydog

exit $?
