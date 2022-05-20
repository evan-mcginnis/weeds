
mkdir sources
cd sources
wget https://files.pythonhosted.org/packages/5a/56/4682a5118a234d15aa1c8768a528aac4858c7b04d2674e18d586d3dfda04/pycuda-2021.1.tar.gz
export CUDA_INC_DIR=/usr/local/cuda-10.2/include
python3 configure.py  --cuda-root=/usr/local/cuda-10.2
sudo sh -c "make install"
