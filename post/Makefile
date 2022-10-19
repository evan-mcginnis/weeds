#
# P O S T  M A K E F I L E
#
application: review_ui.py

review_ui.py: review.ui
	pyuic5 review.ui -o review_ui.py -x

clean:
	rm *.log.*
	rm -rf ../output/*
