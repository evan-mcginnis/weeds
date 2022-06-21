#
# M U C C O M M U N I C A T O R
#
# Encapsulates the XMPP Communications to other components
#
from time import sleep
from typing import Callable
import sys
import asyncio

import xmpp.protocol
from xmpp import *

# Just for the test routines below
import logging
from getpass import getpass
from argparse import ArgumentParser
#import dns.resolver

import constants

#
# M U C C O M M U N I C A T O R
#
# The communication object for the MUC
# Example of how to use this is in the main routine below
#
class MUCCommunicator():
    def __init__(self, jid: str, nickname: str, jidPassword: str, muc: str, processor: Callable):
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
        self._client = None
        self._callback = None
        self._connection = None
        self._connected = False

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
                    print("From: {} Message [{}]".format(msg.getFrom(), msg.getBody()))
                else:
                    print("Keepalive message from chatroom")
                #self.sendMessage("Response")
        if msg.getType() == "chat":
                print("private: " + str(msg.getFrom()) +  ":" +str(msg.getBody()))

    # This looks a bit awkward, but is here just so a keyboard ctrl-c will interrupt things

    def _StepOn(self,conn):
        """
        Process messages on the connection.
        :param conn: Connection to the XMPP server
        :return:
        """
        try:
            # Process the messages, timing out every so often to run some diagnostics
            conn.Process(constants.PROCESS_TIMEOUT)

            # Check to see if the client is still connected to server
            if conn.isConnected():
                print("Still connected")
            else:
                print("--- Disconnected ---")
                return 0
        except KeyboardInterrupt:
                return 0
        return 1

    def GoOn(self,conn):
        while self._StepOn(conn):
            pass

    def connect(self, process: bool):
        """
        Connect to the xmpp server and join the MUC. This routine will not return.
        :return:
        """
        # Create the client
        self._client = xmpp.protocol.JID(self._jid)

        # Preparare the connection (xxx@conference.weeds.com -> weeds.com
        self._connection = xmpp.Client(self._client.getDomain(), debug=[])

        self._connection.connect()

        self._connection.auth(self._client.getNode(),self._password)


        self._connection.sendInitPresence()

        if self._callback is not None:
            self._connection.RegisterHandler('message', self._callback)
        else:
            self._connection.RegisterHandler('message', self.messageCB)


        self._connection.send(xmpp.Presence(to="{}/{}".format(self._muc,self._nickname)))
        self._connected = True

        # I hate delays, but this allows the connection to settle.
        # If this logic sends a message right away, it tends to hit a not connected exception
        time.sleep(5)

        if process:
            self.sendMessage("{} beginning to process messages".format(self._nickname))
            self.GoOn(self._connection)
            # This won't be executed until the processing loop has a keyboard interrupt
            self.sendMessage("{} stopping message processing".format(self._nickname))
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
            id = self._connection.send(stranza)
        else:
            raise ConnectionError(constants.MSG_NOT_CONNECTED)
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
