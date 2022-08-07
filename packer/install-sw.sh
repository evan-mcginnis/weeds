
install_if_missing () {
  for package in "$*"
  do
    dpkg-query -l $package
    if [ $? -gt 0 ]; then
      apt install $package -y
    else
      echo "$package already installed"
    fi
    if [ $? -gt 0 ]; then
      echo "$package install failed"
    fi
  done
  return $?
}

version_installed () {
  version=`pip list | grep $0 | awk '{print $2}'`
  return $version
}

echo
echo ---
echo DNS
echo ---
install_if_missing "dnsutils"
# Perform these steps
# https://www.tecmint.com/set-permanent-dns-nameservers-in-ubuntu-debian/

echo
echo ---
echo PIP
echo ---
echo
# Check to see if pip3 is installed
install_if_missing "python3-pip"

python3 -m pip install --upgrade pip
if [ $? -gt 0 ]; then
  echo "pip upgrade failed"
  exit 1
fi

echo
echo ------
echo PYCUDA
echo ------

install_if_missing "ctags"

#pip3 install --global-option=build_ext --global-option="-I/usr/local/cuda/include" --global-option="-L/usr/local/cuda/lib64" pycuda
if [ ! -d pycuda-2021.1 ]; then
  if [ ! -f pycuda-2021.1.tar.gz ]; then
    wget https://files.pythonhosted.org/packages/5a/56/4682a5118a234d15aa1c8768a528aac4858c7b04d2674e18d586d3dfda04/pycuda-2021.1.tar.gz
  fi
  gunzip pycuda-2021.1.tar.gz
  tar xf pycuda-2021.1.tar
fi

cd pycuda-2021.1/
export CUDA_INC_DIR=/usr/local/cuda-10.2/include
export CUDA_ROOT=/usr/local/cuda/bin
export PATH=$PATH:/usr/local/cuda/bin
python3 configure.py  --cuda-root=/usr/local/cuda-10.2
echo "PWD is $PWD"
make install

if [ $? -gt 0 ]; then
  echo "Pycuda install failed"
  exit 1
fi

echo
echo ------------
echo Requirements
echo ------------
pip3 install -r ~/jetson-requirements.txt
#pip3 install -r ~/requirements-jetson.txt

#echo
#echo -----------
#echo P I L L O W
#echo -----------
#
#yes | pip3 install --upgrade pillow
#if [ $? -gt 0 ]; then
#  echo "pillow install failed"
#  exit 1
#fi


#
# Not really used, but pip3 install of requirements file fails if
# this is not done
#

#echo
#echo ---------------
#echo P R O T O B U F
#echo ---------------
#
#yes | pip3 install --upgrade protobuf
#if [ $? -gt 0 ]; then
#  echo "protobuf install failed"
#  exit 1
#fi

##1. Dependencies
#
echo
echo -----------------------
echo D E P E N D E N C I E S
echo -----------------------
install_if_missing "liblapack-dev"
install_if_missing "gfortran"

#dpkg-query -l  liblapack-dev
#if [ $? gt 0 ]; then
#  sudo apt-get -y install liblapack-dev
#else
#  echo "libapack-dev already installed"
#fi
#
#dpkg-query -l  gfortran
#if [ $? gt 0 ]; then
#  sudo apt-get -y install gfortran
#else
#  echo "gfortran already installed"
#fi


#sudo apt-get install python3-pip
#yes | pip3 install -U pip
#if [ $? -gt 0 ]; then
#  echo "pillow install failed"
#  exit 1
#fi
#yes | pip3 install Cython numpy
#if [ $? -gt 0 ]; then
#  echo "cython and numpy install failed"
#  exit 1
#fi

#2. Scipy
echo
echo ----------
echo S C I P Y
echo ----------
#sudo apt-get update
#sudo apt-get install -y build-essential libatlas-base-dev
#sudo pip install scipy

install_if_missing "build-essential" "libatlas-base-dev"

# Warning -- experiment
#version=$(version_installed "scipy")
#
#if [ "$version" = "1.3.3" ]; then
#  echo
#    echo "scipy already installed"
#  else
#    echo scipy is version $version
#    echo wget https://github.com/scipy/scipy/releases/download/v1.3.3/scipy-1.3.3.tar.gz
#    echo tar -xzvf scipy-1.3.3.tar.gz scipy-1.3.3
#    echo cd scipy-1.3.3/
#    # This gets around the coredump problem if you don't have the CORETYPE specified.
#    # https://stackoverflow.com/questions/65631801/illegal-instructioncore-dumped-error-on-jetson-nano
#    echo OPENBLAS_CORETYPE=ARMV8 python3 setup.py install --user
#fi
#
#if [ $? -gt 0 ]; then
#  echo "scipy install failed"
#  exit 1
#fi

# stop experiment

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
#sudo apt-get install -y python3-sklearn
#sudo apt-get install -y libaec-dev libblosc-dev libffi-dev libbrotli-dev libboost-all-dev libbz2-dev
#sudo apt-get install -y libgif-dev libopenjp2-7-dev liblcms2-dev libjpeg-dev libjxr-dev liblz4-dev liblzma-dev libpng-dev libsnappy-dev libwebp-dev libzopfli-dev libzstd-dev

install_if_missing "python3-sklearn"
install_if_missing "libaec-dev" "libblosc-dev" "libffi-dev" "libbrotli-dev" "libboost-all-dev" "libbz2-dev"
install_if_missing "libgif-dev" "libopenjp2-7-dev" "liblcms2-dev" "libjpeg-dev" "libjxr-dev" "liblz4-dev" "liblzma-dev" "libpng-dev" "libsnappy-dev" "libwebp-dev" "libzopfli-dev" "libzstd-dev"

#sudo pip3 install imagecodecs
# Should be in the requirements.txt
#sudo pip3 install scikit-image
#sudo pip3 install Cython

# P Y P Y L O N

# Should be in the requirements.txt
#sudo pip3 install pypylon

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
#sudo pip3 install pycuda

#
# X M P P
#
#sudo pip3 install xmpppy

# Required upgrades
sudo pip3 install six --upgrade

install_if_missing "ntp" "ntpdate" "ntpstat"

#
# P Y L O N
#

if [ ! -d /opt/pylon ]; then
  mkdir /opt/pylon
  gunzip ~/pylon_6.2.0.21487_aarch64_setup.tar.gz
  tar xf ~/pylon_6.2.0.21487_aarch64_setup.tar
  tar -C /opt/pylon -xzf ./pylon_*.tar.gz
  chmod 755 /opt/pylon
else
  echo "Pylon already installed"
fi

#git clone http://git.code.sf.net/p/linuxptp/code linuxptp
#cd linuxptp
#make

