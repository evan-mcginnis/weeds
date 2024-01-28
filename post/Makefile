#
# P O S T  M A K E F I L E
#

application: review_ui.py

review_ui.py: review.ui
	pyuic5 review.ui -o review_ui.py -x


clean:
	rm -f *.log
	rm -f imbalance.csv

# All imbalance techniques

imbalance:
	python ./imbalance.py -d ../jetson/training-from-dataframe.csv -ini ../jetson/standalone.ini -lg ../jetson/standalone-logging.ini -a ALL --classifier ALL
