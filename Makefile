heuristic:
	python weeds.py -i input2 -o output -a ndi -t "(130,0)" -v

development:	
	python weeds.py -i input2 -o output -a ndi -t "(130,0)" -df training.csv -l -sc -v -d distance

lr:	
	python weeds.py -i input2 -o output -a ndi -t "(130,0)" -df training.csv -l -sc -v

knn:	
	python weeds.py -i input2 -o output -a ndi -t "(130,0)" -df training.csv -k -sc -v

decision:	
	python weeds.py -i input2 -o output -a ndi -t "(130,0)" -df training.csv -dt -sc -v

forest:	
	python weeds.py -i input2 -o output -a ndi -t "(130,0)" -df training.csv -f -sc -v

gradient:	
	python weeds.py -i input2 -o output -a ndi -t "(130,0)" -df training.csv -g -sc -v

gradient2:	
	python weeds.py -i input2 -o output -a ndi -t "(130,0)" -df training-doubled.csv -g -sc -v

treatment:
	python weeds.py -i input2 -o output -a ndi -t "(130,0)" -df training.csv -l -sc -v -sp -d location

treatment2:
	python weeds.py -i input2 -o output -a ndi -t "(130,0)" -df training-doubled.csv -l -sc -v -sp -d location

crop-images:	
	python weeds.py -i input2 -o output -a ndi -t "(130,0)" -df training.csv -k 

