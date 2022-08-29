#!/bin/sh

echo Provision AWS
if [ ! -d .aws ]; then
	mkdir .aws
	cp ~/config.aws ./.aws/config
	cp ~/credentials.aws ./.aws/credentials
fi

exit $?
