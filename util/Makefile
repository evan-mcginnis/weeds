#
# U T I L I T Y
#

TRAINING?=training.csv
application: color_ui.py

color_ui.py: color.ui
	pyuic5 color.ui -o color_ui.py -x

default:
	@echo UTILITIES
	@echo need trarget training-file

clean:
	@rm -f results.csv
	@rm -f color.csv

BEGINALTITUDE?=0
ENDALTITUDE?=5
TECHNIQUE?=lr
BEGINDATE?="2019-01-18"
ENDDATE?="2019-01-19"


training-file:
	python create-training.py -ini ../jetson/standalone.ini -lg ../jetson/standalone-logging.ini -ba $(BEGINALTITUDE) -ea $(ENDALTITUDE) -bd $(BEGINDATE) -ed $(ENDDATE) -m $(TECHNIQUE) -o $(TRAINING)

LEN?=100

# The MAGNITUDE will control the number of axes required for the plot.  Pass in 0, for instance to force a single axis
# The default is to determine how may are required
ifeq ($(MAGNITUDE),)
_MAGNITUDE =
else
_MAGNITUDE = -m $(MAGNITUDE)
endif

plot-for-document:
	test -n "$(X)"
	test -n "$(Y)"
	test -n "$(X2)"
	test -n "$(Y2)"
	test -n "$(INPUT)"
	test -n "$(COLOR)"
	test -n "$(SUBJECT)"
	test -n "$(LEN)"
	$(eval _INPUT = `cygpath -w $(INPUT)`)
	python plot.py -x $(X) -y $(Y) -x2 $(X2) -y2 $(Y2) -i $(_INPUT) -s $(SUBJECT) -l logging.ini -c $(COLOR) -b 0 -b 1 -b 2 -t band --length $(LEN) $(_MAGNITUDE)