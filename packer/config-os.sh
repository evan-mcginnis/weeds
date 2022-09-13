#!/bin/sh

# Set up jumbo frames

mtu=`cat /sys/class/net/eth0/mtu`
if [ $mtu = 1500 ]; then
  echo "Configure eth0 for jumbo frames"
  ifconfig eth0 mtu 9000
  sysctl -w net.core.rmem_max=4096000
  # https://linuxhint.com/how-to-change-mtu-size-in-linux/
  echo "post-up /sbin/ifconfig eth0 mtu 9000 up" >> /etc/network/interfaces
else
  echo "Interface is already $mtu"
fi

echo "Setup for NTP"

# https://ubuntu.com/server/docs/network-dhcp
echo "Setup for DHCP"
sudo systemctl restart isc-dhcp-server.service
exit $?

# This is a client configuration -- putting it here so I don't forget

echo "Configure DNS"
apt install resolvconf
localDNS = `grep "212.40" /etc/resolvconf/resolv.conf.d/head`
if [ ]; then
  echo nameserver 169.254.212.40 >> /etc/resolvconf/resolv.conf.d/head
fi

