#
# C A M E R A  U T I L I T Y
#
#
from pypylon import pylon

tl_factory = pylon.TlFactory.GetInstance()
devices = tl_factory.EnumerateDevices()
for device in devices:
    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateDevice(device))
    print(device.GetFriendlyName())
    camera.Open()

    camera.DeviceInfo().
    # to get consistant results it is always good to start from "power-on" state
    camera.UserSetSelector = "Default"
    camera.UserSetLoad.Execute()

info = camera.DeviceInfo


