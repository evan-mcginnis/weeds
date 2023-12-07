#!/bin/sh

sudo apt-get update
sudo apt install python3-pip -y
version_matplotlib=`python3 -c "import matplotlib; print(matplotlib.__version__)"`

echo
echo ---------------------
echo matplotlib $version_matplotlib
echo ---------------------

echo
echo -----
echo P I P
ecch -----
python3 -m pip install -y --upgrade pip
if [ $? -gt 0 ]:
  echo "pip upgrade failed"
  exit 1
fi

echo
echo -----------
echo P I L L O W
echo -----------

python3 -m pip install -y --upgrade pillow
if [ $? -gt 0 ]:
  echo "pillow install failed"
  exit 1
fi
#1. Dependencies

#sudo apt-get install liblapack-dev gfortran
#sudo apt-get install python3-pip
#sudo pip3 install -U pip
#sudo pip3 install Cython numpy
#
##2. Scipy
#
#wget https://github.com/scipy/scipy/releases/download/v1.3.3/scipy-1.3.3.tar.gz
#tar -xzvf scipy-1.3.3.tar.gz scipy-1.3.3
#cd scipy-1.3.3/
#python3 setup.py install --user
#
##3. Tiff
#
#wget https://download.osgeo.org/libtiff/tiff-4.1.0.tar.gz
#tar -xzvf tiff-4.1.0.tar.gz
#cd tiff-4.1.0/
#./configure
#make
#sudo make install
##4. Scikit-image
#
#sudo apt-get install python3-sklearn
#sudo apt-get install libaec-dev libblosc-dev libffi-dev libbrotli-dev libboost-all-dev libbz2-dev
#sudo apt-get install libgif-dev libopenjp2-7-dev liblcms2-dev libjpeg-dev libjxr-dev liblz4-dev liblzma-dev libpng-dev libsnappy-dev libwebp-dev libzopfli-dev libzstd-dev
#sudo pip3 install imagecodecs
#sudo pip3 install scikit-image
