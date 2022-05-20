
python3 -m pip install --upgrade pip
if [ $? -gt 0 ]; then
  echo "pip upgrade failed"
  exit 1
fi

echo
echo -----------
echo P I L L O W
echo -----------

yes | pip3 install --upgrade pillow
if [ $? -gt 0 ]; then
  echo "pillow install failed"
  exit 1
fi

#
# Not really used, but pip3 install of requirements file fails if
# this is not done
#

echo
echo ---------------
echo P R O T O B U F 
echo ---------------

yes | pip3 install --upgrade protobuf
if [ $? -gt 0 ]; then
  echo "protobuf install failed"
  exit 1
fi

##1. Dependencies
#
echo
echo -----------------------
echo D E P E N D E N C I E S
echo -----------------------
sudo apt-get -y install liblapack-dev gfortran
sudo apt-get install python3-pip
yes | pip3 install -U pip
if [ $? -gt 0 ]; then
  echo "pillow install failed"
  exit 1
fi
yes | pip3 install Cython numpy
if [ $? -gt 0 ]; then
  echo "cython and numpy install failed"
  exit 1
fi

#2. Scipy
echo
echo ----------
echo S C I P Y
echo ----------
sudo apt-get update
sudo apt-get install -y build-essential libatlas-base-dev
sudo pip install scipy
#wget https://github.com/scipy/scipy/releases/download/v1.3.3/scipy-1.3.3.tar.gz
#tar -xzvf scipy-1.3.3.tar.gz scipy-1.3.3
#cd scipy-1.3.3/
# This gets around the coredump problem if you don't have the CORETYPE specified.
# https://stackoverflow.com/questions/65631801/illegal-instructioncore-dumped-error-on-jetson-nano
#OPENBLAS_CORETYPE=ARMV8 python3 setup.py install --user

if [ $? -gt 0 ]; then
  echo "scipy install failed"
  exit 1
fi
#3. Tiff
#
#wget https://download.osgeo.org/libtiff/tiff-4.1.0.tar.gz
#tar -xzvf tiff-4.1.0.tar.gz
#cd tiff-4.1.0/
#./configure
#make
#sudo make install
#if [ $? -gt 0 ]; then
#  echo "TIFF install failed"
#  exit 1
#fi
#4. Scikit-image
#
sudo apt-get install -y python3-sklearn
sudo apt-get install -y libaec-dev libblosc-dev libffi-dev libbrotli-dev libboost-all-dev libbz2-dev
sudo apt-get install -y libgif-dev libopenjp2-7-dev liblcms2-dev libjpeg-dev libjxr-dev liblz4-dev liblzma-dev libpng-dev libsnappy-dev libwebp-dev libzopfli-dev libzstd-dev
#sudo pip3 install imagecodecs
sudo pip3 install scikit-image
sudo pip3 install Cython

# P Y P Y L O N

sudo pip3 install pypylon

# Test

#sudo - H pip3 install joblib numpy scipy
#sudo - H pip3 install scikit - learn--index - url https: //piwheels.org/simple
#
#sudo apt - get install python - dev libfreetype6 - dev
#sudo apt - get install libfreetype6 - dev
#sudo ln - s / usr / include / freetype2 / freetype / /usr/include / freetype
#sudo apt - get install libfontconfig1 - dev
#sudo - H pip3 install scikit - image

#
# N U M P Y  C U D A
#
sudo pip3 install pycuda



