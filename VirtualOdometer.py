import threading
from time import sleep

from Odometer import Odometer
from MUCCommunicator import MUCCommunicator
import constants
import logging
import time

class VirtualOdometer(Odometer):
    def __init__(self, muc: MUCCommunicator, speed: float):
        """
        A simulated odometer
        :param muc: Multi-user chat communicator
        :param speed: Speed of movement in meters per second
        """
        self._speedInCMPerSecond = speed * 100
        self._secondsPerCM = 1 / self._speedInCMPerSecond
        self._log = logging.getLogger(__name__)

        self._start = 0
        self._elapsed_milliseconds = 0
        self._elapsed = 0
        self._xmpp = muc
        self.log = logging.getLogger(__name__)

    def connect(self) -> bool:
        # Connect to the chatroom
        self._xmpp.connect()
        return True

    def disconnect(self):
        return True

    def diagnostics(self):
        self._connected = False
        return True, "Odometer diagnostics passed"

    def registerCallback(self,callback):
        self._callback = callback

    def registerMUC(self, muc: MUCCommunicator):
        self._xmpp = muc

    def start(self):

        # 4 KPH is 111.111 cm per second
        time.sleep(3)
        #sleepTime = self._secondsPerCM
        sleepTime = 0.5
        i = 0
        while True:
            print("Sleep for {0} i = {1}".format(sleepTime,i))
            time.sleep(sleepTime)
            self._xmpp.send_message(mto=constants.ROOM_ODOMETRY,
                                    mbody="VirtualOdometer::start " + str(i),
                                    mtype='groupchat')
            #self._xmpp.process(forever=True, timeout=5)
            #self._xmpp.process(forever=True, timeout=1)
            i += 1
            # Apply the treatment
            # An alternative here would be to communicate with a treatment thread
            #self._callback()


if __name__ == '__main__':

    def startupCommunication(muc: MUCCommunicator):
        muc.process()

    messageNumber = 0
    def process(msg):
        global messageNumber
        print("Process {}".format(messageNumber))
        messageNumber += 1

    # Just for the test routines below
    import logging
    from getpass import getpass
    from argparse import ArgumentParser
    import dns.resolver
    from threading import Thread

    # Setup the command line arguments.
    parser = ArgumentParser()

    # Output verbosity options.
    parser.add_argument("-q", "--quiet", help="set logging to ERROR",
                        action="store_const", dest="loglevel",
                        const=logging.ERROR, default=logging.INFO)
    parser.add_argument("-de", "--debug", help="set logging to DEBUG",
                        action="store_const", dest="loglevel",
                        const=logging.DEBUG, default=logging.INFO)

    parser.add_argument('-d', '--dns', action="store", required=True, help="DNS server address")

    # JID and password options.
    parser.add_argument("-j", "--jid", dest="jid",
                        help="JID to use")
    parser.add_argument("-p", "--password", dest="password",
                        help="password to use")
    parser.add_argument("-r", "--room", dest="room",
                        help="MUC room to join")
    parser.add_argument("-n", "--nick", dest="nick",
                        help="MUC nickname")

    args = parser.parse_args()

    # Setup logging.
    logging.basicConfig(level=args.loglevel,
                        format='%(levelname)-8s %(message)s')

    if args.jid is None:
        args.jid = input("Username: ")
    if args.password is None:
        args.password = getpass("Password: ")
    if args.room is None:
        args.room = input("MUC room: ")
    if args.nick is None:
        args.nick = input("MUC nickname: ")

    # Force resolutions to come from a server that has the entries we want
    my_resolver = dns.resolver.Resolver()
    my_resolver.nameservers = [args.dns]

    # Setup the MUCBot and register plugins. Note that while plugins may
    # have interdependencies, the order in which you register them does
    # not matter.
    xmpp = MUCCommunicator(args.jid, args.password, args.room, args.nick, process)
    xmpp.startup(5)
    #initializeXMPP(xmpp)
    xmpp.register_plugin('xep_0030') # Service Discovery
    xmpp.register_plugin('xep_0045') # Multi-User Chat
    xmpp.register_plugin('xep_0199') # XMPP Ping

    odometer = VirtualOdometer(xmpp, 10)
    odometer.connect()
    #xmpp.connect()
    xmpp.process()
    odometer.connect()
    odometer.start()

    # Keep track of the threads we create
    threads = list()

    communicator = threading.Thread(target=startupCommunication, args=(xmpp,))
    threads.append(communicator)
    communicator.start()
