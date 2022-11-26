#
# I N T E L  R E A L S E N S E
#
# Install the intel realsense drivers and environment
#

# The instructions are here:
# https://dev.intelrealsense.com/docs/nvidia-jetson-tx2-installation

# Server key
sudo apt-key adv --keyserver keys.gnupg.net --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE || sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE

# Add repo
sudo add-apt-repository "deb https://librealsense.intel.com/Debian/apt-repo bionic main" -u

# install the SDK
sudo apt-get install -y librealsense2-utils
sudo apt-get install -y librealsense2-dev

# Compile from source, as Intel does not supply the python for arm64
# Required for build
sudo apt-get install -y libssl-dev

# Clone the repo
git clone https://github.com/IntelRealSense/librealsense.git
cd librealsense
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=true -DBUILD_PYTHON_BINDINGS=bool:true -DPYTHON_EXECUTABLE=/usr/bin/python3
make -j4
sudo make install


exit 0