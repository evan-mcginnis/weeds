;
; BIND data file for local loopback interface
;
$TTL	604800
@	IN	SOA	weeds.com. root.weeds.com. (
			     45		; Serial
			 604800		; Refresh
			  86400		; Retry
			2419200		; Expire
			 604800 )	; Negative Cache TTL
;
@	IN	NS	ns.weeds.com.
@	IN	A	169.254.212.40	
@	IN	AAAA	::1
ns	IN	A	169.254.212.40
rio	IN	A	169.254.212.39
vmware	IN	A	169.254.212.31
jetson	IN	A	169.254.212.40
;conference IN      A    169.254.212.31
_xmpp-client._tcp.weeds.com.  3600 IN  SRV 0 5 5222 jetson.weeds.com.
_xmpp-server._tcp.weeds.com.  3600 IN  SRV 0 5 5269 jetson.weeds.com.
_xmpp-server._tcp.conference.weeds.com. 3600 IN  SRV 0 5 5269 jetson.weeds.com.

