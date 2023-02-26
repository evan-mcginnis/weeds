#
# Message Queue Communicator
#
import zmq
from typing import Callable
import threading

import constants

TYPE_CLIENT = 0
TYPE_SERVER = 1

class MQCommunicator:

    def __init__(self, **kwargs):
        """
        Connectivity to the zeromq system.  Type is based on the presence or absence of keywords

        :param kwargs: SERVER=<ip> and PORT=<int> for client keywords, and
        PORT=<int> for server
        """
        self._context = zmq.Context()
        self._type = TYPE_SERVER
        self._message = ""
        self._lock = threading.Lock()
        self._exiting = False

    @property
    def message(self) -> str:
        """
        The current message to be sent
        :return: current message
        """
        self._lock.acquire()
        msg = self._message
        self._lock.release()
        return msg

    @message.setter
    def message(self, theMessage: str):
        """
        Sets the current message to the specified string. This string will be given in response to the command
        :param theMessage:
        """
        self._lock.acquire()
        self._message = theMessage
        self._lock.release()

    def sendMessage(self):
        """
        Publish a single message
        """
        try:
            self._socket.send_string("{}".format(self._message))
        except zmq.error.ZMQError as err:
            if not self._exiting:
                print("{}".format(err))

    def sendSpecificMessage(self, message: str):
        self._socket.send_string(message)

    def receiveMessage(self) -> str:
        try:
            data = self._socket.recv().decode("utf-8")
        except zmq.error.ZMQError as err:
            if not self._exiting:
                print("{}".format(err))
            data = None
        return data

    def stop(self):
        self._exiting = True
        self._context.destroy()
        self._socket.close()

class ServerMQCommunicator(MQCommunicator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        try:
            self._port = kwargs[constants.KEYWORD_PORT]
        except KeyError:
            raise ValueError("Expected {} defined as keyword".format(constants.KEYWORD_PORT))

        self._socket = self._context.socket(zmq.REP)
        self._socket.bind("tcp://*:%s" % self._port)

class ClientMQCommunicator(MQCommunicator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._callback = None

        self._server = kwargs[constants.KEYWORD_SERVER]
        self._port = kwargs[constants.KEYWORD_PORT]
        self._type = TYPE_CLIENT
        self._socket = self._context.socket(zmq.REQ)
        self._messagesToProcess = 100
        self._poller = None
        self._connected = False

    @property
    def connect(self) -> bool:
        return self._connected

    @property
    def messages(self) -> int:
        return self._messagesToProcess

    @messages.setter
    def messages(self, numberOfMessages: int):
        self._messagesToProcess = numberOfMessages

    @property
    def callback(self) -> Callable:
        """
        The callback for each message received
        :return:
        """
        return self._callback

    @callback.setter
    def callback(self, callback: Callable):
        """
        Set the callback function for each message
        :param callback: a function taking a single string argument
        """
        self._callback = callback

    def connect(self):
        self._socket = self._context.socket(zmq.REQ)
        self._socket.connect("tcp://%s:%s" % (self._server, self._port))

        self._poller = zmq.Poller()
        self._poller.register(self._socket, zmq.POLLIN)
        self._connected = True

    def disconnect(self):
        self._poller.unregister(self._socket)
        self.connect()

    def start(self, command: str):
        """
        Receive the publications from the server, and call back for each message,
        """

        # self._socket.setsockopt_string(zmq.SUBSCRIBE, self._topic)

        while self._messagesToProcess:
            self._messagesToProcess -= 1
            # Issue the command repeatedly and forward the response to the callback registered
            self.message = command
            self.sendMessage()
            string = self._socket.recv()
            self._callback(string)

    def sendMessageAndWaitForResponse(self, message: str, timeout: int) -> (bool, str):
        """
        Sends the specified message and returns the response
        :param message: message to send
        :param timeout: timeout in milliseconds
        :return: (bool, str)
        """
        response = ""
        responded = False

        # Send the specified message
        self.message = message
        self.sendMessage()

        # Wait for the response
        socks = dict(self._poller.poll(timeout))
        if socks:
            if socks.get(self._socket) == zmq.POLLIN:
                response = self._socket.recv(zmq.NOBLOCK)
                responded = True
            else:
                responded = False

        return responded, response

    def waitForServer(self, retries: int) -> bool:
        responded = False

        # Loop and accept messages from both channels, acting accordingly
        while retries and not responded:
            retries -= 1
            self.message = constants.COMMAND_PING
            self.sendMessage()
            poller = zmq.Poller()
            poller.register(self._socket, zmq.POLLIN)
            socks = dict(poller.poll(1000))
            if socks:
                if socks.get(self._socket) == zmq.POLLIN:
                    print("got message : ".format(self._socket.recv(zmq.NOBLOCK)))
                    responded = True
            else:
                poller.unregister(self._socket)
                self.connect()
        return responded

if __name__ == "__main__":
    from argparse import ArgumentParser
    import time
    import keyboard
    from Messages import OdometryMessage
    # Setup the command line arguments.
    parser = ArgumentParser()

    parser.add_argument("-t", "--type", help="client or server", action="store", required=True, choices=["client", "server"])
    parser.add_argument("-a", "--address", help="Address of server", action="store", required=False)
    parser.add_argument("-p", "--port", help="Port", action="store", required=False)
    parser.add_argument("-d", "--distance", help="Output just the distance", action="store_true", required=False, default=False)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-r", "--repeat", help="Repeat", action="store", type=int, required=False, default=1)
    group.add_argument("-c", "--continuous", help="Receive continuous messages (Client only)", action="store_true", default=False, required=False)

    args = parser.parse_args()

    currentDistance = 0

    def onKeyPress():
        print("Key pressed\n")

    def messageCallback(message: str):
        print(message)

    if args.type == "client":
        communicator = ClientMQCommunicator(SERVER=args.address, PORT=args.port)
        communicator.connect()
        communicator.callback = messageCallback
        if not args.continuous:
            repeat = args.repeat
        else:
            repeat = args.continuous

        (serverIsResponding, response) = communicator.sendMessageAndWaitForResponse(constants.COMMAND_PING, 5000)
        #keyboard.on_press_key("p", onKeyPress())

        if serverIsResponding:
            while repeat:
                if not args.continuous:
                    repeat -= 1
                (serverIsResponding, response) = communicator.sendMessageAndWaitForResponse(constants.COMMAND_ODOMETERY, 1000)
                #serverIsResponding = communicator.waitForServer(5)
                if serverIsResponding:
                    if args.distance:
                        odometryMessage = OdometryMessage(raw=response)
                        if keyboard.is_pressed("p"):
                            onKeyPress()
                        print(odometryMessage.totalDistance, end='\r')
                    else:
                        print("[{}]".format(response))
                    # communicator.start(constants.COMMAND_ODOMETERY)
                else:
                    print("Server did not respond.")
        else:
            print("Server did not respond within 5 seconds")

    elif args.type == "server":
        communicator = ServerMQCommunicator(PORT=args.port)
        pulse = 0
        while True:
            message = communicator.receiveMessage()
            if message == constants.COMMAND_ODOMETERY:
                print("Received request for odometry")
                pulse += 1
                response = OdometryMessage()
                response.pulses = pulse
                response.speed = 4
                response.distance = 40
                response.totalDistance = 40
                communicator.message = response.formMessage()
                communicator.sendMessage()
            if message == constants.COMMAND_PING:
                print("Received Ping")
                communicator.message = constants.RESPONSE_ACK
                communicator.sendMessage()

