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

BEGINALTITUDE?=0
ENDALTITUDE?=5
TECHNIQUE?=lr
BEGINDATE?="2019-01-18"
ENDDATE?="2019-01-19"


training-file:
	python create-training.py -ini ../jetson/standalone.ini -lg ../jetson/standalone-logging.ini -ba $(BEGINALTITUDE) -ea $(ENDALTITUDE) -bd $(BEGINDATE) -ed $(ENDDATE) -m $(TECHNIQUE) -o $(TRAINING)
