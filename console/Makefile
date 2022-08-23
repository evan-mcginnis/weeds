application: console.py dialog_init.py

console.py: console.ui
	pyuic5 console.ui -o console.py -x

dialog_init.py: dialog-init.ui
	pyuic5 dialog-init.ui -o dialog_init.py -x

requirements:
	pipreqs

requirements-install:
	pip3 install -r requirements.txt

clean:
	rm console.log*


