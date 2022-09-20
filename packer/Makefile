rio:
	packer build rio-admin.json
	packer build rio-user.json

jetson:
	packer build jetson-admin.json
	packer build jetson-software.json

jetson-left-admin:
	packer build -var-file=variables-left.json -on-error=abort jetson-left-admin.json

xavier-left-admin:
	packer build -var-file=variables-xavier-left.json -on-error=abort jetson-left-admin.json

xavier-right-admin:
	packer build -var-file=variables-xavier-right.json -on-error=abort jetson-right-admin.json

# Install the items needed by the operational software
xavier-left:
	packer build -var-file=variables-xavier-left.json -on-error=abort jetson-software.json

xavier-right:
	#packer build -var-file=variables-xavier-right.json -on-error=abort jetson-software.json
	packer build -var-file=variables-xavier-right.json -on-error=abort jetson-software-right.json

jetson-left:
	packer build -var-file=variables-left.json jetson-software.json

jetson-users:
	packer build -var-file=variables-left.json jetson-users.json
	packer build -var-file=variables-right.json jetson-users.json

iron-chef: rio-software
	packer build -var-file=variables-iron-chef.json iron-chef.json

daq: daq-software  post-software
	packer build -var-file=variables-daq.json daq.json

post-software:
	cd ..; tar cf post.tar post

jetson-software: 
	cd ..; tar cf jetson.tar jetson lib http

rio-software:
	cd ..; tar cf rio.tar rio lib http

daq-software:
	cd ..; tar cf rio.tar rio lib http

distribute-software-tucson: jetson-software rio-software post-software
	packer build -var-file=variables-left.json jetson.json
	packer build -var-file=variables-right.json jetson.json
	packer build -var-file=variables-daq.json daq.json

distribute-software-yuma: jetson-software rio-software post-software
	packer build -var-file=variables-xavier-left.json jetson.json
	packer build -var-file=variables-xavier-right.json jetson.json
	packer build -var-file=variables-daq.json daq.json


distribute-rio: rio-software
	packer build -var-file=variables-rio.yuma.json rio.json

