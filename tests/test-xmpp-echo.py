import asyncio
import logging

from slixmpp import ClientXMPP
import sys
import asyncio

import dns.resolver
import argparse

#my_resolver = dns.resolver.Resolver()
#my_resolver.nameservers = ['169.254.212.31']

#answer = my_resolver.resolve('rio.weeds.local')
#print(answer.response)

# This is specific to the windows platform to avoid the NotImplemented error
# Probably not required on Linux
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


class EchoBot(ClientXMPP):

    def __init__(self, jid, password):
        ClientXMPP.__init__(self, jid, password)

        self.add_event_handler("session_start", self.session_start)
        self.add_event_handler("message", self.message)

        # If you wanted more functionality, here's how to register plugins:
        # self.register_plugin('xep_0030') # Service Discovery
        # self.register_plugin('xep_0199') # XMPP Ping

        # Here's how to access plugins once you've registered them:
        # self['xep_0030'].add_feature('echo_demo')

    def session_start(self, event):
        self.send_presence()
        self.get_roster()

        # Most get_*/set_* methods from plugins use Iq stanzas, which
        # are sent asynchronously. You can almost always provide a
        # callback that will be executed when the reply is received.

    def message(self, msg):
        if msg['type'] in ('chat', 'normal'):
            msg.reply("Thanks for sending\n%(body)s" % msg).send()


if __name__ == '__main__':
    # Ideally use optparse or argparse to get JID,
    # password, and log level.
    parser = argparse.ArgumentParser("XMPP Echo Utility")

    parser.add_argument('-d', '--dns', action="store", required=True, help="DNS server address")

    arguments = parser.parse_args()

    # Force resolutions to come from a server that has the entries we want
    my_resolver = dns.resolver.Resolver()
    my_resolver.nameservers = [arguments.dns]
    answer = my_resolver.resolve('rio.weeds.local')
    print(answer.response)
    logging.basicConfig(level=logging.DEBUG,
                        format='%(levelname)-8s %(message)s')

    xmpp = EchoBot('rio@weeds.local', 'weeds')
    xmpp.connect()
    xmpp.process()
