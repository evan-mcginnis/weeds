#
#
sudo apt install bind9
sudo apt install dnsutils
# Copy named.conf db.169 and db.weeds.local under /etc/bind
# Copy named.conf.local under /etc/bind
sudo systemctl restart bind9.service

