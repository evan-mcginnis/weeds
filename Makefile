lr:	
	python weeds.py -i input2 -o output -a ndi -t "(130,0)" -df lr.csv -lr -sc

knn:	
	python weeds.py -i input2 -o output -a ndi -t "(130,0)" -df lr.csv -k -sc

crop-images:	
	python weeds.py -i input2 -o output -a ndi -t "(130,0)" -df lr.csv -k -x
