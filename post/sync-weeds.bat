@echo 
@echo Synch files with weeding systems
@echo
rem Make sure we get the correct ssh
PATH=c:\cygwin64\bin;%PATH%
cd c:\tmp
rsync -avz -e ssh -r weeds@169.254.212.40:output .
rsync -avz -e ssh -r weeds@169.254.212.50:output .
rsync -avz -e ssh -r weeds@169.254.212.35:output .

