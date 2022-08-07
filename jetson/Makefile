#
# M A K E F I L E
#
clean:
	rm -f *.log*
	rm -rf ../output/*

requirements:
	pipreqs

grab-arm:
	OPENBLAS_CORETYPE=ARMv8 python3 weeds.py --grab

grab:
	python3 weeds.py -o ../output/ -lg logging.ini --grab