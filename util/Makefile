#
# U T I L I T Y
#

include ../common.mk

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


# Combine the results from several days to yield one classification file
broccoli-season:
	$(eval INPUT = $(BROCCOLI_CLASSIFICATIONS))
	test -n "$(INPUT)"
	test -n "$(OUTPUT)"
	$(eval _INPUT = `cygpath -w $(INPUT)`)
	$(eval _OUTPUT = `cygpath -w $(OUTPUT)`)
	python combine.py -i $(_INPUT) --output $(_OUTPUT)

cantaloupe-season:
	$(eval INPUT = $(CANTALOUPE_CLASSIFICATIONS))
	test -n "$(INPUT)"
	test -n "$(OUTPUT)"
	$(eval _INPUT = `cygpath -w $(INPUT)`)
	$(eval _OUTPUT = `cygpath -w $(OUTPUT)`)
	python combine.py -i $(_INPUT) --output $(_OUTPUT)

shape-season:
	$(eval INPUT = $(SHAPE_CLASSIFICATIONS))
	test -n "$(OUTPUT)"
	$(eval _INPUT = `cygpath -w $(INPUT)`)
	$(eval _OUTPUT = `cygpath -w $(OUTPUT)`)
	python combine.py -i $(_INPUT) --output $(_OUTPUT)

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

lbp-figure:
	test -n "$(LEFT)"
	test -n "$(RIGHT)"
	test -n "$(OUTPUT)"
	$(eval _LEFT = `cygpath -w $(LEFT)`)
	$(eval _RIGHT = `cygpath -w $(RIGHT)`)
	$(eval _OUTPUT = `cygpath -w $(OUTPUT)`)
	python lbp-graph.py -l $(_LEFT) -r $(_RIGHT) -o $(_OUTPUT)

calibration-plot:
	$(eval _CORRECTED = `cygpath -w $(CORRECTED)`)
	$(eval _UNCORRECTED = `cygpath -w $(UNCORRECTED)`)
	$(eval _SCRATCH = `cygpath -w $(SCRATCH)`)
	$(eval _OUTPUT = `cygpath -w $(OUTPUT)`)
	python plot-calibration.py -c $(_CORRECTED) -u $(_UNCORRECTED) --scratch $(_SCRATCH) --output $(_OUTPUT)
