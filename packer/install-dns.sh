#
#
sudo apt install bind9 -y
sudo apt install dnsutils -y
# Copy named.conf db.169 and db.weeds.com under /etc/bind
# Copy named.conf.local under /etc/bind
sudo systemctl restart bind9.service

