#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys,time
import xmpp
from argparse import ArgumentParser
import dns.resolver

def sendMessageToGroup(server, user, password, room, message):

    jid = xmpp.protocol.JID(user)
    user = jid.getNode()
    client = xmpp.Client(server)
    connection = client.connect(secure=False)
    if not connection:
        print('connection failed')

        sys.exit(1)

    auth = client.auth(user, password)
    if not auth:
        print('authentication failed')
        sys.exit(1)

    # Join a room by sending your presence to the room
    client.send(xmpp.Presence(to="%s/%s" % (room, user)))

    msgObj = xmpp.protocol.Message(room, message)
    #Set message type to 'groupchat' for conference messages
    msgObj.setType('groupchat')

    client.send(msgObj)

    # some older servers will not send the message if you disconnect immediately after sending
    time.sleep(1)

    client.disconnect()

# Setup the command line arguments.
parser = ArgumentParser()

# Output verbosity options.
# parser.add_argument("-q", "--quiet", help="set logging to ERROR",
#                     action="store_const", dest="loglevel",
#                     const=logging.ERROR, default=logging.INFO)
# parser.add_argument("-de", "--debug", help="set logging to DEBUG",
#                     action="store_const", dest="loglevel",
#                     const=logging.DEBUG, default=logging.INFO)

parser.add_argument('-d', '--dns', action="store", required=True, help="DNS server address")

# JID and password options.
parser.add_argument("-j", "--jid", dest="jid", help="JID to use")
parser.add_argument("-p", "--password", dest="password", help="password to use")
parser.add_argument("-r", "--room", dest="room", help="MUC room to join")
parser.add_argument("-n", "--nick", dest="nick", help="MUC nickname")

arguments = parser.parse_args()

#
# D N S
#

if arguments.dns is not None:
    # Force resolutions to come from a server that has the entries we want
    my_resolver = dns.resolver.Resolver()
    my_resolver.nameservers = [arguments.dns]
    answer = my_resolver.resolve('vmware.weeds.local')
    print(answer.response)

sendMessageToGroup("vmware.weeds.local",arguments.jid, arguments.password, arguments.room, "This is a test")