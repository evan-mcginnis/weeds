#
if [ -f /etc/ejabberd/ejabberd.yml ]; then
	echo "Ejabberd already installed"
	exit 0
else
	# Copy the configuration file up
	# /etc/ejabberd/ejabberd.yml
	echo sudo apt-get -y install ejabberd
	echo ejabberdctl register admin weeds.com greydog
	echo ejabberdctl register rio weeds.com greydog
	echo ejabberdctl register jetson weeds.com greydog
	echo ejabberdctl register console weeds.com greydog
fi
exit $?


# To remove this
# sudo apt-get remove ejabberd

