@echo 
@echo Update tablet software 
@echo
rem Make sure we get the correct ssh
PATH=c:\cygwin64\bin;%PATH%
cd c:\Users\weeds\weeds
git pull > c:\tmp\update.log
