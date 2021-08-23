#!/bin/sh -x

if [ -f "/home/weeds/miniconda3/etc/profile.d/conda.sh" ]; then
  . "/home/weeds/miniconda3/etc/profile.d/conda.sh"
  export PATH="/home/weeds/miniconda3/bin:$PATH"
else
  export PATH="/home/weeds/miniconda3/bin:$PATH"
fi

pythonVersion=$(python -c 'import platform; print(platform.python_version())')
PREFIX=./miniconda3

if [[ "$pythonVersion" == "3.9.5" ]]; then
	echo "Correct version of python installed. No action taken"
else
	wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
	sh Miniconda3-latest-Linux-x86_64.sh -b
	$PREFIX/bin/conda init
fi

source ~weeds/.bashrc

pip install numpy
pip install nidaqmx
exit $? 

