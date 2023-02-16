#
# P O S T  M A K E F I L E
#
application: review_ui.py select_image_set.py

review_ui.py: review.ui
	pyuic5 review.ui -o review_ui.py -x

select_image_set.py: select_image_set.ui
	pyuic5.exe select_image_set.ui -o select_image_set.py -x

clean:
	rm *.log.*
	rm -rf ../output/*
