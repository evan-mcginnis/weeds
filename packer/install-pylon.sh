#/bin/sh

# This is not yet tested, but this is the logic for installing
# the basler software on the jetson.

gunzip pylon_6.2.0.21487_aarch64_setup.tar.gz
tar xf pylon_6.2.0.21487_aarch64_setup.tar
sudo tar -C /opt/pylon -xzf ./pylon_*.tar.gz
sudo chmod 755 /opt/pylon

