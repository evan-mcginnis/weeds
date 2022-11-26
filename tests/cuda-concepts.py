#import pycuda.driver as drv
import datetime
import sys

import pycuda.autoinit
import numpy as np

from pycuda import gpuarray
from pycuda.elementwise import ElementwiseKernel

import pycuda.driver as cuda

#drv.init()

#host_data = np.array([1,2,3,4,5],dtype=np.int8)
host_data = np.random.randint(0,10,size=(2880,2160), dtype=np.int8)



def bygpuarray():
    # Copy to the GPU
    startTime = datetime.datetime.now()
    device_data = gpuarray.to_gpu(host_data)
    finishTime = datetime.datetime.now()
    computeTime = finishTime - startTime
    print("Copy to GPU with gpuarray: {}".format(computeTime))



    # # The first time this is done, we need to compile this computation for the GPU,
    # # so the time taken will be longer
    device_data_x2 = 2 * device_data
    # device_data_x2 = 2 * device_data
    # device_data_x2 = 2 * device_data
    #
    startTime = datetime.datetime.now()
    device_data_x2 = 2 * device_data
    finishTime = datetime.datetime.now()
    computeTime = finishTime - startTime
    print("compute with gpuarray: {}".format(computeTime))
    #
    startTime = datetime.datetime.now()
    host_data_x2 = device_data_x2.get()
    finishTime = datetime.datetime.now()
    computeTime = finishTime - startTime
    print("get from GPU with gpuarray: {}".format(computeTime))
    #
    #
    print(host_data_x2)

    # Allocate the memory on the GPU
    device_data = cuda.mem_alloc(host_data.nbytes)
    # Copy the numpy array to its counterpart on the GPU.
    # This needs to be a contiguous array to work properly
    startTime = datetime.datetime.now()
    cuda.memcpy_htod(device_data, np.ascontiguousarray(host_data))
    finishTime = datetime.datetime.now()
    computeTime = finishTime - startTime
    print("Copy to GPU with memcpy: {}".format(computeTime))

    device_data.free()

def elementwise():
    kernelCIVE = ElementwiseKernel(
        "float *red, float *green, float *blue, float *out",
        "out[i] = red[i] * 0.441 - 0.881 * green[i] + 0.385 * blue[i] + 18.78745;",
        "kernelCIVE"
    )
    #host_data = np.random.randint(0,10,size=(2880,2160), dtype=np.int8)
    red = np.random.randn(2880,2160).astype(np.float32)
    green = np.random.randn(2880,2160).astype(np.float32)
    blue = np.random.randn(2880,2160).astype(np.float32)
#    host_data = np.float32(np.random.random(5000))
    _deviceRed = gpuarray.to_gpu(red)
    _deviceGreen = gpuarray.to_gpu(green)
    _deviceBlue = gpuarray.to_gpu(blue)
    _deviceCIVE = gpuarray.empty_like(_deviceRed)
    #
    # #Elementwise
    # So the kernel is downloaded and compiled before we take timing scored
    kernelCIVE(_deviceRed, _deviceGreen, _deviceBlue, _deviceCIVE)
    startTime = datetime.datetime.now()
    kernelCIVE(_deviceRed, _deviceGreen, _deviceBlue, _deviceCIVE)
    finishTime = datetime.datetime.now()
    computeTime = finishTime - startTime
    from_device = _deviceCIVE.get()
    print("Compute with elementwise kernel: {} ({} - {})".format(computeTime, startTime, finishTime))
    print(from_device)

elementwise()
