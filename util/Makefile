#
# U T I L I T Y
#

TRAINING?=training.csv

default:
	@echo UTILITIES
	@echo need trarget training-file

training-file:
	python create-training.py -ini ../jetson/standalone.ini -lg ../jetson/standalone-logging.ini -a 0 -o $(TRAINING)
