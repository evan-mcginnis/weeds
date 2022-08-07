#
# Set up the hostname
sudo hostnamectl set-hostname vmware
# Make this change in /etc/hosts

# Files to copy
# ejabberd.yml
# TODO: PTP Configuration

# Prevent that memory hog from starting
cd /etc/init.d
mv salt-call salt-call.bak
mv salt-minion salt-minion.bak
mv salt-proxy salt-proxy.bak
mv niminionagent niminionagent.bak

cd /usr/local/natinst/share/NIWebServer