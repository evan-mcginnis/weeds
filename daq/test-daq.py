import time

import nidaqmx as ni

system = ni.system.System.local()
system.driver_version
devices = system.devices
for device in system.devices:
    print(device)

channels = ni.system.physical_channel.PhysicalChannel("Mod5")
print(channels)
with ni.Task() as task:
    task.do_channels.add_do_chan('Mod4/port0/line3')
    task.do_channels.add_do_chan('Mod4/port0/line4')
    task.do_channels.add_do_chan('Mod4/port0/line2')
    #task.do_channels.add_do_chan('Mod4/port0')

    print('1 Channel 3 Sample Write: ')
    print(task.write([True,True,True]))
    time.sleep(10)
    print(task.write([False,False,False]))
    time.sleep(2)

# with ni.Task() as task:
#     task.ao_channels.add_ao_voltage_chan("Mod4/ai0")
#     task.read(number_of_samples_per_channel=2)



# import PyDAQmx as daq
# import numpy
#
# analog_input = daq.Task()
# read = daq.int32()
# data = numpy.zeros((1000,), dtype=numpy.float64)
#
# # DAQmx Configure Code
# analog_input.CreateAIVoltageChan("Dev1/ai0","",daq.DAQmx_Val_Cfg_Default,-10.0,10.0,daq.DAQmx_Val_Volts,None)
# analog_input.CfgSampClkTiming("",10000.0,daq.DAQmx_Val_Rising,daq.DAQmx_Val_FiniteSamps,1000)
#
# # DAQmx Start Code
# analog_input.StartTask()
#
# # DAQmx Read Code
# analog_input.ReadAnalogF64(1000,10.0,daq.DAQmx_Val_GroupByChannel,data,1000,daq.byref(read),None)
#
# print "Acquired %d points"%read.value
