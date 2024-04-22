application: UI435.py

UI435.py: UI435.ui
	pyuic5 UI435.ui -o UI435.py -x

requirements:
	pipreqs

requirements-install:
	pip3 install -r requirements.txt

run:
	python capture-435.py -l ..\jetson\logging.ini -o ..\..\test-images

clean:
	rm *.log*


