from xmpp import *

class Sample:
    def __init__(self, jid: str, jid_password: str, muc: str):
        self.jid = jid
        self.password = jid_password
        self.muc = muc

    def messageCB(self,conn,msg):
        print(msg.getType())
        if msg.getType() == "groupchat":
                print(str(msg.getFrom()) +": "+  str(msg.getBody()))
        if msg.getType() == "chat":
                print("private: " + str(msg.getFrom()) +  ":" +str(msg.getBody()))

    def StepOn(self,conn):
        try:
            conn.Process(1)
        except KeyboardInterrupt:
                return 0
        return 1

    def GoOn(self,conn):
        while self.StepOn(conn):
            pass

    def send_muc(self, msg: str, cl):
        stranza = "<message to='{0}' type='groupchat'><body>{1}</body></message>".format(self.muc, msg)
        cl.send(stranza)
        #cl.send(Message("odometry@conference.weeds.com", msg))

    def receive_muc(self):

        client=xmpp.protocol.JID(self.jid)

        cl = xmpp.Client(client.getDomain(), debug=[])

        cl.connect()

        cl.auth(client.getNode(),self.password)


        cl.sendInitPresence()

        cl.RegisterHandler('message', self.messageCB)

        room = self.muc
        print("Joining " + room)

        cl.send(xmpp.Presence(to="{}/{}".format(room,"/jetson")))
        self.send_muc("Hello from jetson", cl)

        self.GoOn(cl)

sample = Sample("jetson@weeds.com", "greydog", "odometry@conference.weeds.com")
sample.receive_muc()
