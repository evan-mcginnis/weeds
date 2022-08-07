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

# install required libraries
pip install numpy
pip install nidaqmx

# AWS access
pip install boto3

# Just a placeholder for now -- the xmpp library is in the python-packages
# directory.  This line will not work as written
uncompress xmpppy-0.7.1.tar.gz
tar xf xmppy-0.7.1.tar

python -m pip install ./xmpppy-0.7.1


exit $? 

