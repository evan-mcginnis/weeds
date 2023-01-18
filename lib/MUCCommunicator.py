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

import traceback

import xmpp.protocol
from xmpp import *

# Just for the test routines below
import logging
from getpass import getpass
from argparse import ArgumentParser
#import dns.resolver
import threading

from WeedExceptions import XMPPServerUnreachable, XMPPServerAuthFailure
import constants

from statemachine import StateMachine, State
from statemachine.exceptions import TransitionNotAllowed

class MUCState(StateMachine):
    new = State('New', initial=True)
    connected = State('Connected')
    disconnected = State('Disconnected')
    pending = State('Pending')

    initialized = new.to(pending)
    toPending = disconnected.to(pending)
    toConnected = pending.to(connected)
    toDisconnected = connected.to(disconnected)
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
        self._reconnectOnSendFailure = False
        self._id = 0
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
        self._currentOperation = constants.Status.QUIESCENT
        self._state = MUCState()

        try:
            self._timeout = kwargs[constants.KEYWORD_TIMEOUT]
        except KeyError as key:
            self._timeout = constants.PROCESS_TIMEOUT_LONG

    @property
    def state(self) -> MUCState:
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
        # with self._lock:
        #     isConnected = self._connected
        # return isConnected
        return self._connected

    @connected.setter
    def connected(self, connectionState: bool):
        # with self._lock:
        #     self._connected = connectionState
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
            # self._log.debug("Processing messages with timeout: {}".format(self._timeout))
            try:
                conn.Process(self._timeout)
            except xmpp.protocol.SystemShutdown:
                self._log.fatal("XMPP System shutdown")
                self._connected = False
                self._state.toDisconnected()
                return 0
            except Exception as e:
                self._log.error("Exception in message processing")
                self._log.error("Raw: {}".format(e))
                self._log.error("{}".format(traceback.format_exc()))
                return 0


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

    def streamErrorHandler(self, connection, error):
        self._log.fatal("Stream error encountered")

    def disconnectHandler(self):
        self._log.critical("Disconnected from {}".format(self._muc))

        # There may have already been an error noticed in send, so don't reconnect in that case
        if not self._connected:
            self._connected = False
            #self._reconnectOnSendFailure = True
            while not self._connected:
                self._log.debug("Reconnecting to chatroom")
                self.connectToChatroom()
        else:
            self._log.warning("Will not reconnect to chatroom")

        # if not self._connection.reconnectAndReauth():
        #     self._log.fatal("Unable to recover connection")
        #     self._state = constants.Status.EXIT_FATAL
        # else:
        #     if not self._connection.isConnected:
        #         self._log.fatal("Reconnect indicates success, but not connected")
        #         self._state = constants.Status.EXIT_FATAL
        #     else:
        #         self._log.warning("Reconnected to chatroom")
        #         self._connected = True

    def getOccupants(self):
        try:
            self._occupants = xmpp.features.discoverItems(self._connection, self._muc)
        except UnicodeDecodeError as unicode:
            self._log.error("Unable to fetch occupants")
            self._log.error(unicode)

    def connectToChatroom(self):
        """
        Connect to chatroom and announce presence.
        """
        # This code is a duplicate of what is below, but this is the routine that should be used.

        self._client = xmpp.protocol.JID(self._jid)

        # Prepare the connection (xxx@conference.weeds.com -> weeds.com
        #self._connection = xmpp.Client(self._client.getDomain(), debug=[])
        self._connection = xmpp.Client(self._client.getDomain(), debug=debug)

        # TODO: This is wrong -- the code should lookup the host based on the SRV record to get both the hostname and port number
        #self._connection.connect(server=('jetson-right.weeds.com', 5222))
        connectionType = self._connection.connect(server=(self._server, 5222))
        self._log.debug("Connected with type: [{}]".format(connectionType))

        if connectionType != 'tls':
            self._log.error("Unable to connect to XMPP server with a TLS")
            return

        self._connection.auth(self._client.getNode(),self._password)


        self._connection.sendInitPresence()

        if self._callback is not None:
            self._connection.RegisterHandler('message', self._callback)
        else:
            self._connection.RegisterHandler('message', self.messageCB)

        self._connection.RegisterHandler('error', self.streamErrorHandler,xmlns=NS_STREAMS)

        self._log.debug("Sending presence {}/{}".format(self._muc, self._nickname))
        self._connection.send(xmpp.Presence(to="{}/{}".format(self._muc,self._nickname)))
        self._connected = True

    def connect(self, process: bool, occupants = False, processCallback: Callable = None, processData = None):
        """
        Connect to the xmpp server and join the MUC. This routine will not return.
        :return:
        """
        shouldRetry = False

        if self.state.is_new:
            self._log.debug("Connecting for first time")
            self.state.initialized()
        elif self.state.is_disconnected:
            self._log.debug("Reconnecting to chatroom")
            self.state.toPending()

        # Create the client
        self._client = xmpp.protocol.JID(self._jid)

        # Preparare the connection (xxx@conference.weeds.com -> weeds.com
        self._connection = xmpp.Client(self._client.getDomain(), debug=[])

        # TODO: This is wrong -- the code should lookup the host based on the SRV record to get both the hostname and port number
        #self._connection.connect(server=('jetson-right.weeds.com', 5222))
        connectionType = self._connection.connect(server=(self._server, 5222))
        self._log.debug("Connected with type: [{}]".format(connectionType))

        # The only connection type considered a success is TLS.  If the server is not up, this will be ''
        if connectionType != 'tls':
            self._log.warning("Unable to connect to XMPP server: {}/5222".format(self._server))
            self._connected = False
            raise XMPPServerUnreachable("Unable to connect with TLS")

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
        self._currentOperation = constants.Status.RUNNING
        self._state.toConnected()
        self._log.debug("Connected to {}".format(self._muc))

        # I hate delays, but this allows the connection to settle.
        time.sleep(4)

        if process:
            self.processing = True

            # As this routine never returns, we need a mechanism to call back the requester just prior to
            # entering the message processing loop.  This is used by the UI to show the status of the connection.

            if processCallback is not None:
                processCallback(processData)

            #self.sendMessage("{} beginning to process messages".format(self._nickname))
            self.GoOn(self._connection)
            # This won't be executed until the processing loop has a keyboard interrupt
            #self.sendMessage("{} stopping message processing".format(self._nickname))
        return

    def processMessages(self):
        """
        Process messages for the chatroom -- note that this routine will never return.
        :param communicator: The chatroom communicator
        """
        self._log.info("Connecting to chatroom")
        processing = True

        while processing:
            try:
                # This method should never return unless something went wrong
                self.connect(True)
                self._log.debug("Connected and processed messages, but encountered errors")
            except XMPPServerUnreachable:
                self._log.warning("Unable to connect and process messages.  Will retry.")
                time.sleep(5)
                processing = True
            except XMPPServerAuthFailure:
                self._log.fatal("Unable to authenticate using parameters")
                processing = False

    def disconnect(self):
        raise NotImplementedError

    def sendMessage(self, msg: str) -> int:
        """
        Sends a message to the already connected XMPP server/MUC.
        Raises ConnectionError if not connected to server
        :param msg:
        :return:
        """

        self._id += 1
        # Send the message if we are still connected
        if self._connected and self._connection.isConnected():
            # Original stanza
            #stranza = "<message to='{0}' type='groupchat'><body>{1}</body></message>".format(self._muc, msg)
            stranza = "<message to='{0}' type='groupchat' id='{1}'><body>{2}</body></message>".format(self._muc, self._id, msg)
            #self._log.debug("Sending {}".format(stranza))
            try:
                self._connection.send(stranza)
            except ConnectionResetError as reset:
                self._log.error("---- Connection reset error encountered ----")
            except IOError as io:
                self._log.error("I/O Error encountered. Typically this means that the server kicked us out of the MUC")
                # Reconnect to the chatroom, as the disconnect handler scheme seems not to work as I want it to

                #self.connectToChatroom()
                #self._log.debug("Connected to chatroom")
                time.sleep(2)
                if self._currentOperation == constants.Status.EXIT_FATAL:
                    self._log.error("XMPP connection cannot be recovered. -- E X I T I N G --")
                    sys.exit(-1)
            except Exception as ex:
                self._log.fatal("Unknown error {}".format(ex))
                sys.exit(-1)

        else:
            self._log.error("Not connected to MUC")
            #self.connectToChatroom()
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
