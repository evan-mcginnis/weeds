import os

#os.add_dll_directory(os.path.join(os.environ['CUDA_PATH'], 'bin'))

import pycuda.driver as drv

drv.init()

print("Detected {} CUDA devices".format(drv.Device.count()))
for i in range(drv.Device.count()):
    device = drv.Device(i)
    print("Device {}: {}".format(i, device.name()))
