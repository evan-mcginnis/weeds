#
# W E E D S  P R O J E C T
#
#

OS := $(shell uname)

#
# Executable names
PYTHON = python

ifeq ($(OS),Darwin)
	PYTHON = python3
endif

# This file should exist on a jetson.
# If we don't have that bit before the python command, you get a coredump when you import

ifeq (,$(wildcard /sys/module/tegra_fuse/parameters/tegra_chip_id))
	PYTHON = OPENBLAS_CORETYPE=ARMv8 python3
endif

PYLINT=pylint

#
# Defaults if they are not specified on the make line
#
# Log configuration
LOG?=info-logging.yaml
# output directory
OUTPUT?=output
# input image set
INPUT?=input2
# Training data
TRAINING?=training-521.csv
# Decorations on the output images
DECORATIONS?=none
# Machine learning algorithm to use
ML?=lr
# Vegetation index algorithm
INDEX?=ndi
# Parameter Selection
PARAMETERS?=all-parameters.csv

# Clean this up a bit, reducing the ML algorithm to one parameter
ALGFLAG=-k

ifeq ($(ML),knn)
	ALGFLAG = -k
endif
ifeq ($(ML), lr)
	ALGFLAG = -l
endif
ifeq ($(ML), gradient)
	ALGFLAG = -g
endif
ifeq ($(ML), forest)
	ALGFLAG = -f
endif
ifeq ($(ML), decision)
	ALGFLAG = -dt
endif

# Indicate that the system should produce treatment plans as an image
TREATMENT?=-sp

weeds:
	$(PYTHON) weeds.py -i $(INPUT) -o $(OUTPUT) -a $(INDEX) -t "(130,0)" -df $(TRAINING) $(ALGFLAG) -sc -v -lg $(LOG) -se $(PARAMETERS)

treatment:
	$(PYTHON) weeds.py -i $(INPUT) -o $(OUTPUT) -a $(INDEX) -t "(130,0)" -df $(TRAINING) $(ALGFLAG) -sc -v -lg $(LOG) $(TREATMENT) -d location

# Clean up various files
clean:
	rm weeds.log

lint:
	$(PYLINT) weeds.py

hsv:
	$(PYTHON) view-hsv.py -i $(OUTPUT)/original-11.jpg

# L A T E X
thesis:
	pdflatex thesis.tex

# Various targets that are probably not needed anymore

heuristic:
	$(PYTHON) weeds.py -i $(INPUT) -o $(OUTPUT) -a ndi -t "(130,0)" -v

development:	
	$(PYTHON) weeds.py -i $(INPUT) -o $(OUTPUT) -a ndi -t "(130,0)" -df $(TRAINING) -l -sc -v -d $(DECORATIONS)

minimal:	
	$(PYTHON) weeds.py -i $(INPUT) -o $(OUTPUT) -a ndi -t "(130,0)" -df $(TRAINING) -l -sc -v -d none


lr-hsv:	
	$(PYTHON) weeds.py -i $(INPUT)4 -o $(OUTPUT) -a ndi -t "(130,0)" -df $(TRAINING) -l -sc -v

lr:	
	$(PYTHON) weeds.py -i $(INPUT) -o $(OUTPUT) -a ndi -t "(130,0)" -df $(TRAINING) -l -sc -v

lr3:
	$(PYTHON) weeds.py -i $(INPUT)3 -o $(OUTPUT) -a ndi -t "(130,0)" -df $(TRAINING) -l -sc -v

knn:	
	$(PYTHON) weeds.py -i $(INPUT) -o $(OUTPUT) -a ndi -t "(130,0)" -df $(TRAINING) -k -sc -v

decision:	
	$(PYTHON) weeds.py -i $(INPUT) -o $(OUTPUT) -a ndi -t "(130,0)" -df $(TRAINING) -dt -sc -v

forest:	
	$(PYTHON) weeds.py -i $(INPUT) -o $(OUTPUT) -a ndi -t "(130,0)" -df $(TRAINING) -f -sc -v

gradient:	
	$(PYTHON) weeds.py -i $(INPUT) -o $(OUTPUT) -a ndi -t "(130,0)" -df $(TRAINING) -g -sc -v


treatment2:
	$(PYTHON) weeds.py -i $(INPUT) -o $(OUTPUT) -a ndi -t "(130,0)" -df training-doubled.csv -l -sc -v -sp -d location

crop-images:	
	$(PYTHON) weeds.py -i $(INPUT) -o $(OUTPUT) -a ndi -t "(130,0)" -df $(TRAINING) -k

proposal.bbl: paperpile.bib
	biber proposal

proposal-sources:
	biber proposal

proposal: proposal.tex proposal.bbl
	pdflatex proposal.tex