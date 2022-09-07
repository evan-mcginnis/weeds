#
# M U C C O M M U N I C A T O R
#
# Encapsulates the XMPP Communications to other components
#
from time import sleep
from typing import Callable
import sys
import asyncio
import threading

import xmpp.protocol
from xmpp import *

# Just for the test routines below
import logging
from getpass import getpass
from argparse import ArgumentParser
#import dns.resolver
import threading

import constants

#
# M U C C O M M U N I C A T O R
#
# The communication object for the MUC
# Example of how to use this is in the main routine below
#
class MUCCommunicator():
    def __init__(self, server: str, jid: str, nickname: str, jidPassword: str, muc: str, processor: Callable, presence: Callable, **kwargs):
        """
        The MUC communication class.
        :param jid: The JID used i.e., rio@weeds.com
        :param nickname:  The nickname of the JID in the room
        :param jidPassword: The password for the JID
        :param muc:  The full name of the room i.e., odometry@conference.weeds.com
        :param processor: The message processing routine. If this is None, the default processor is used
        """
        self._jid = jid
        self._nickname = nickname
        self._password = jidPassword
        self._muc = muc
        self._server = server
        self._client = None
        self._callback = processor
        self._callbackPresence = presence
        self._connection = None
        self._connected = False
        #self._log = logging.getLogger(__name__)
        self._log = logging.getLogger(threading.current_thread().getName())
        self._occupants = list()
        self._lock = threading.Lock()

        self._processing = False
        self._state = constants.Status.QUIESCENT

        try:
            self._timeout = kwargs[constants.KEYWORD_TIMEOUT]
        except KeyError as key:
            self._timeout = constants.PROCESS_TIMEOUT_LONG

    @property
    def state(self) -> constants.Status:
        return self._state

    @property
    def processing(self) -> bool:
        return self._processing

    @processing.setter
    def processing(self, newProcessingState: bool):
        if not newProcessingState:
            self._log.warning("Stop processing MUC messages")
        self._processing = newProcessingState

    def refresh(self):
        return
        # if self._connected:
        #     try:
        #         self._occupants = xmpp.features.discoverItems(self._connection, self._muc)
        #         for occupant in self._occupants:
        #             print("Occupant {}:".format(occupant.get("jid")))
        #     except Exception as e:
        #         self._log.error("Encountered XMPP error in refresh")
        #         self._log.error("Raw: {}".format(e))

    def occupantExited(self, occupant: str) -> bool:
        found = False
        for i in range(len(self._occupants)):
            if self._occupants[i]['name'] == occupant:
                del self._occupants[i]
                found = True
                break
        return found

    def occupantEntered(self, occupant: str, jid: str) -> bool:
        inserted = True
        occupantRecord = {'name': occupant, 'jid': jid}
        self._occupants.append(occupantRecord)

        return inserted

    @property
    def occupants(self):
        return self._occupants

    @property
    def callback(self):
        return self._callback

    @callback.setter
    def callback(self, callback: Callable):
        self._callback = callback

    @property
    def connected(self) -> bool:
        # Read lock
        with self._lock:
            isConnected = self._connected
        return isConnected

    @connected.setter
    def connected(self, connectionState: bool):
        with self._lock:
            self._connected = connectionState

    @property
    def connection(self) -> xmpp.Client:
        return self._connection

    @property
    def muc(self):
        # Read lock
        with self._lock:
            mucName = self._muc
        return mucName

    @muc.setter
    def muc(self, mucName: str):
        with self._lock:
            self._muc = mucName

    def diagnostics(self) -> ():
        """
        Perform diagnostics on the MUC connection and room
        :return:
        """
        diagnosticResult = False
        diagnosticText = constants.MSG_NO_PROBLEM_FOUND

        if not self._connection.isConnected():
            diagnosticResult = False
            diagnosticText = constants.MSG_NOT_CONNECTED
        else:
            diagnosticResult = True
            diagnosticText = constants.MSG_NO_PROBLEM_FOUND

        return (diagnosticResult, diagnosticText)

    # The default callback is useful if the process is only interested in publishing
    # and does not concern itself with what room occupants have to say

    def messageCB(self,conn,msg: xmpp.protocol.Message):
        #print(msg.getType())
        if msg.getType() == "groupchat":
                #print(str(msg.getFrom()) +": "+  str(msg.getBody()))
                body = msg.getBody()
                # Check if this is a real message and not just an empty keep-alive message
                if body is not None:
                    self._log.debug("From: {} Message [{}]".format(msg.getFrom(), msg.getBody()))
                else:
                    self._log.debug("Keepalive message from chatroom")
                #self.sendMessage("Response")
        if msg.getType() == "chat":
                self._log.error("Unexpected private message: " + str(msg.getFrom()) +  ":" +str(msg.getBody()))

    def _presenceCB(self, conn, presence):
        self._log.debug("Presence: {}".format(presence.getFrom().getStripped()))

    # This looks a bit awkward, but is here just so a keyboard ctrl-c will interrupt things

    def _StepOn(self,conn):
        """
        Process messages on the connection.
        :param conn: Connection to the XMPP server
        :return:
        """
        try:
            # Process the messages, timing out every so often to run some diagnostics
            self._log.debug("Processing messages")
            try:
                conn.Process(self._timeout)
            except Exception as e:
                self._log.error("Exception in message processing")
                self._log.error("Raw:{}".format(e))

            if not self.processing:
                self._log.debug("No longer processing messages")
                return 0

            # Check to see if the client is still connected to server
            if not conn.isConnected():
                self._log.error("Disconnected from chatroom.")
                #conn.reconnectAndReauth()
                return 0

            # if conn.isConnected():
            #     print("Still connected")
            # else:
            #     print("--- Disconnected ---")
            #     return 0
        except KeyboardInterrupt:
                return 0
        return 1

    def GoOn(self,conn):
        while self._StepOn(conn):
            pass

    def disconnectHandler(self):
        self._log.critical("Disconnected from {}".format(self._muc))
        self._connected = False
        if not self._connection.reconnectAndReauth():
            self._log.fatal("Unable to recover connection")
            self._state = constants.Status.EXIT_FATAL
        else:
            self._log.warning("Reconnected to MUC")
            self._connected = True

    def connect(self, process: bool, occupants = False):
        """
        Connect to the xmpp server and join the MUC. This routine will not return.
        :return:
        """
        # Create the client
        self._client = xmpp.protocol.JID(self._jid)

        # Preparare the connection (xxx@conference.weeds.com -> weeds.com
        self._connection = xmpp.Client(self._client.getDomain(), debug=[])

        # TODO: This is wrong -- the code should lookup the host based on the SRV record to get both the hostname and port number
        #self._connection.connect(server=('jetson-right.weeds.com', 5222))
        self._connection.connect(server=(self._server, 5222))

        self._connection.auth(self._client.getNode(),self._password)


        self._connection.sendInitPresence()

        if self._callback is not None:
            self._connection.RegisterHandler('message', self._callback)
        else:
            self._connection.RegisterHandler('message', self.messageCB)

        self._log.debug("Sending presence {}/{}".format(self._muc, self._nickname))
        self._connection.send(xmpp.Presence(to="{}/{}".format(self._muc,self._nickname)))

        if occupants:
            # This line is troublesome on the jetson
            self._occupants = xmpp.features.discoverItems(self._connection, self._muc)

        self._connection.RegisterDisconnectHandler(self.disconnectHandler)
        # for i in xmpp.features.discoverItems(self._connection, self._muc):
        #     (ids, features) = xmpp.features.discoverInfo(self._connection, i.get("jid"))
        #     print("Occupant {}:".format(i.get("jid")))
        #     # if NS_MUC in features:
        #     #     print("Occupant {}:".format(i.get("jid")))

        # If this logic sends a message right away, it tends to hit a not connected exception
        if self._callbackPresence is not None:
            self._connection.RegisterHandler('presence', self._callbackPresence)

        # else:
        #     self._connection.RegisterHandler('presence', self._presenceCB)

        self.connected = True
        self._state = constants.Status.RUNNING

        # I hate delays, but this allows the connection to settle.
        time.sleep(4)

        if process:
            self.processing = True
            #self.sendMessage("{} beginning to process messages".format(self._nickname))
            self.GoOn(self._connection)
            # This won't be executed until the processing loop has a keyboard interrupt
            #self.sendMessage("{} stopping message processing".format(self._nickname))
        return

    def disconnect(self):
        raise NotImplementedError

    def sendMessage(self, msg: str) -> int:
        """
        Sends a message to the already connected XMPP server/MUC.
        Raises ConnectionError if not connected to server
        :param msg:
        :return:
        """

        id = 0
        # Send the message if we are still connected
        if self._connected and self._connection.isConnected():
            stranza = "<message to='{0}' type='groupchat'><body>{1}</body></message>".format(self._muc, msg)
            #print("Sending {}".format(stranza))
            try:
                id = self._connection.send(stranza)
            except ConnectionResetError as reset:
                self._log.fatal("---- Connection reset error encountered ----")
            except IOError as io:
                self._log.fatal("I/O Error encountered. Typically this means that the server kicked us out of the MUC")
                # Let the reconnect handler do its magic
                time.sleep(2)

        else:
            pass
            # Restructuring things a bit -- the disconnect handler should do this
            # self._log.error("Not connected to MUC")
            # self._connection.reconnectAndReauth()
            # if self._connection.isConnected():
            #     self._log.warning("Reestablished connection, but something is wrong")
            # else:
            #     self._log.fatal("Tried to reconnect, but failed")
            #     raise ConnectionError(constants.MSG_NOT_CONNECTED)
        return id

if __name__ == '__main__':
    import threading

    messageNumber = 0
    def process(msg):
        global messageNumber
        print("Process {}".format(messageNumber))
        messageNumber += 1

    # Setup the command line arguments.
    parser = ArgumentParser()

    # Output verbosity options.
    parser.add_argument("-q", "--quiet", help="set logging to ERROR",
                        action="store_const", dest="loglevel",
                        const=logging.ERROR, default=logging.INFO)
    parser.add_argument("-de", "--debug", help="set logging to DEBUG",
                        action="store_const", dest="loglevel",
                        const=logging.DEBUG, default=logging.INFO)

    parser.add_argument('-d', '--dns', action="store", required=False, help="DNS server address")

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
    # print("Using DNS: " + args.dns)
    # my_resolver = dns.resolver.Resolver()
    # my_resolver.nameservers = [args.dns]
    # answer = my_resolver.resolve('rio.weeds.com')
    # print(answer.response)

    odometry = MUCCommunicator(args.jid, args.nick, args.password, args.room, process)

    def processMessages():
        # Connect to the XMPP server and start processing XMPP stanzas.
        # The only boolean parameter here indicates that this should process messages, not connect and return
        odometry.connect(False)

    def generate():
        while True:
            sleep(5)
            odometry.sendMessage("Message from generator")

    # Start two threads, one to generate messages, one to process
    service = threading.Thread(target = processMessages())
    service.start()

    generator = threading.Thread(target = generate())
    generator.start()

    service.join()
    generator.join()
