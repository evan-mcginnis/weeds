#
# E J A B B E R D
#
#
#install_if_required "ejabberd"

if [ -f /etc/ejabberd/ejabberd.yml ]; then
	echo "Ejabberd already installed"
	exit 0
else
	# Copy the configuration file up
	# /etc/ejabberd/ejabberd.yml
	sudo apt-get -y install ejabberd
	echo "Customize ejabberd"
	sleep 5
	ejabberdctl stop
	cp ~/ejabberd.yml /etc/ejabberd.yml
	echo "Restart ejabberd"
	ejabberdctl start
	# Wait for the server to restart
	sleep 10
	ejabberdctl register admin weeds.com greydog
	ejabberdctl register rio weeds.com greydog
	ejabberdctl register jetson weeds.com greydog
	ejabberdctl register console weeds.com greydog
	ejabberdctl register debug weeds.com greydog
fi
exit $?


# To remove this
# sudo apt-get remove ejabberd

