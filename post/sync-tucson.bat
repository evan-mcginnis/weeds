@echo 
@echo Synch files with jetson systems
@echo
rem Make sure we get the correct ssh
PATH=c:\cygwin64\bin;%PATH%
cd c:\tmp
rsync -avz -e ssh -r weeds@jetson-right.weeds.com:output .
rsync -avz -e ssh -r weeds@jetson-left.weeds.com:output .
rsync -avz -e ssh -r weeds@iron-chef.weeds.com:output .

