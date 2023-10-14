#
# U T I L I T Y
#

TRAINING?=training.csv

default:
	@echo UTILITIES
	@echo need trarget training-file

BEGINALTITUDE?=0
ENDALTITUDE?=5
TECHNIQUE?=lr


training-file:
	python create-training.py -ini ../jetson/standalone.ini -lg ../jetson/standalone-logging.ini -ba $(BEGINALTITUDE) -ea $(ENDALTITUDE) -m $(TECHNIQUE) -o $(TRAINING)
