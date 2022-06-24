#
# D I A G N O S T I C S
#

import asyncio
import configparser

import aioxmpp.muc
import aioxmpp.muc.xso

from MUCFramework import MUCFramework, exec_example


class ServerInfo(MUCFramework):
    def __init__(self, jid: str, jidPassword, muc: str, mucPassword):
        super().__init__(jid, jidPassword, muc, mucPassword)


    # def make_simple_client(self):
    #     client = super().make_simple_client()


    async def run_simple_example(self):
        config = await self.client.summon( aioxmpp.MUCClient ).get_room_config( self.muc_jid )
        if config is not None:
            form = aioxmpp.muc.xso.ConfigurationForm.from_xso(config)

            print("name:", form.roomname.value)
            print("description:", form.roomdesc.value)

            print("Moderated?", form.moderatedroom.value)

            print("members only?", form.membersonly.value)

            print("persistent?", form.persistentroom.value)
        else:
            print("Could not get info on room")


if __name__ == "__main__":
    jid = "admin@weeds.com"
    password = "greydog"
    muc = "odometry@conference.weeds.com"
    exec_example(ServerInfo(jid, password, muc, None))