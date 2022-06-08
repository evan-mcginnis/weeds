#
# M A K E F I L E
# 

CALIBRATION_FILE = "basler-1920-computar5mm.p"
IMAGE_DISTORTED = "calibration-images/calibration-01.jpg"
IMAGE_UNDISTORTED = "calibration-images/calibration-01_undistorted.jpg"
IMAGES = "calibration-images"
PERFORMANCE = "performance.csv"

calibration:
	python camera-calibrate.py -i $(IMAGES) -u $(IMAGE_DISTORTED) -c $(CALIBRATION_FILE) -p $(PERFORMANCE)

clean:
	rm $(IMAGE_UNDISTORTED)
	rm $(CALIBRATION_FILE)
	rm $(PERFORMANCE)
