
OS := $(shell uname)

#
# Executable names

ifeq ($(OS),Darwin)
	PYTHON = python3
else
	PYTHON = python
endif
PYLINT=pylint

#
# Defaults if they are not specified on the make line
#
LOG?=info-logging.yaml
OUTPUT?=output
INPUT?=input2
TRAINING?=training.csv
DECORATIONS?=none
ML?=lr
INDEX?=ndi


# Clean this up a bit, reducing the ML algorithm to one parameter
ALGFLAG=-k

ifeq ($(ML),knn)
	ALGFLAG = -k
endif
ifeq ($(ML), lr)
	ALGFLAG = -l
endif

weeds:
	$(PYTHON) weeds.py -i $(INPUT) -o $(OUTPUT) -a $(INDEX) -t "(130,0)" -df $(TRAINING) $(ALGFLAG) -sc -v -lg $(LOG)

# Clean up various files
clean:
	rm weeds.log

lint:
	$(PYLINT) weeds.py

hsv:
	$(PYTHON) view-hsv.py -i $(OUTPUT)/original-11.jpg

heuristic:
	$(PYTHON) weeds.py -i $(INPUT) -o $(OUTPUT) -a ndi -t "(130,0)" -v

development:	
	$(PYTHON) weeds.py -i $(INPUT) -o $(OUTPUT) -a ndi -t "(130,0)" -df $(TRAINING) -l -sc -v -d all

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

gradient:
	$(PYTHON) weeds.py -i $(INPUT) -o $(OUTPUT) -a ndi -t "(130,0)" -df training-doubled.csv -g -sc -v

treatment:
	$(PYTHON) weeds.py -i $(INPUT) -o $(OUTPUT) -a ndi -t "(130,0)" -df $(TRAINING) -l -sc -v -sp -d location

treatment2:
	$(PYTHON) weeds.py -i $(INPUT) -o $(OUTPUT) -a ndi -t "(130,0)" -df training-doubled.csv -l -sc -v -sp -d location

crop-images:	
	$(PYTHON) weeds.py -i $(INPUT) -o $(OUTPUT) -a ndi -t "(130,0)" -df $(TRAINING) -k

