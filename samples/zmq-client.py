#
#   Hello World client in Python
#   Connects REQ socket to tcp://localhost:5555
#   Sends "Hello" to server, expects "World" back
#

import zmq
import time

context = zmq.Context()

#  Socket to talk to server
print("Connecting to hello world server...")
socket = context.socket(zmq.REQ)
socket.connect("tcp://169.254.212.40:5555")

#  Do 10 requests, waiting each time for a response
for request in range(10):
    started = time.time()
    print(f"Sending request {request} ...")
    socket.send_string("Hello")

    #  Get the reply.
    message = socket.recv()
    stopped = time.time()
    print(f"Received reply {request} [ {message} ] in {stopped - started} ms")