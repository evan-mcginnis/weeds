@echo 
@echo Synch files with jetson systems
@echo
rem Make sure we get the correct ssh
PATH=c:\cygwin64\bin;%PATH%
cd c:\tmp
rsync -avz -e ssh -r weeds@xavier-right.weeds.com:output .
rsync -avz -e ssh -r weeds@xavier-left.weeds.com:output .
rsync -avz -e ssh -r weeds@copper-chef.weeds.com:output .

