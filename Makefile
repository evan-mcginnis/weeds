
OS := $(shell uname)

ifeq ($(OS),Darwin)
	PYTHON = python3
else
	PYTHON = python
endif

heuristic:
	$(PYTHON) weeds.py -i input2 -o output -a ndi -t "(130,0)" -v

development:	
	$(PYTHON) weeds.py -i input2 -o output -a ndi -t "(130,0)" -df training.csv -l -sc -v -d distance

lr:	
	$(PYTHON) weeds.py -i input2 -o output -a ndi -t "(130,0)" -df training.csv -l -sc -v

knn:	
	$(PYTHON) weeds.py -i input2 -o output -a ndi -t "(130,0)" -df training.csv -k -sc -v

decision:	
	$(PYTHON) weeds.py -i input2 -o output -a ndi -t "(130,0)" -df training.csv -dt -sc -v

forest:	
	$(PYTHON) weeds.py -i input2 -o output -a ndi -t "(130,0)" -df training.csv -f -sc -v

gradient:	
	$(PYTHON) weeds.py -i input2 -o output -a ndi -t "(130,0)" -df training.csv -g -sc -v

gradient2:	
	$(PYTHON) weeds.py -i input2 -o output -a ndi -t "(130,0)" -df training-doubled.csv -g -sc -v

treatment:
	$(PYTHON) weeds.py -i input2 -o output -a ndi -t "(130,0)" -df training.csv -l -sc -v -sp -d location

treatment2:
	$(PYTHON) weeds.py -i input2 -o output -a ndi -t "(130,0)" -df training-doubled.csv -l -sc -v -sp -d location

crop-images:	
	$(PYTHON) weeds.py -i input2 -o output -a ndi -t "(130,0)" -df training.csv -k 

