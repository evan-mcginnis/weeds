#
# M A K E F I L E
#

default:
	@echo Targets are data, segmented, and resize
	@echo resize requires INPUT and OUTPUT to be passed in

data:
	@echo Copy images
	cp -rf /cygdrive/d/maricopa/corrected/2024-11-15/pivot-iphone/masks /cygdrive/d/maricopa-test/segment/originals
	cp -rf /cygdrive/d/maricopa/corrected/2024-11-15/flood-iphone/masks /cygdrive/d/maricopa-test/segment/originals
	cp -rf /cygdrive/d/maricopa/corrected/2024-11-15/drip-iphone/masks /cygdrive/d/maricopa-test/segment/originals
	@echo Copy masks
	mkdir /cygdrive/d/maricopa-test/segment/originals/images
	cp  /cygdrive/d/maricopa/corrected/2024-11-15/pivot-iphone/*.jpg /cygdrive/d/maricopa-test/segment/originals/images
	cp  /cygdrive/d/maricopa/corrected/2024-11-15/flood-iphone/*.jpg /cygdrive/d/maricopa-test/segment/originals/images
	cp  /cygdrive/d/maricopa/corrected/2024-11-15/drip-iphone/*.jpg /cygdrive/d/maricopa-test/segment/originals/images


segmented:
	python ..\lib\VegetationIndex.py -i d:\maricopa-test\segment\originals -o d:\maricopa-test\segment\train.modified\images -m d:\maricopa-test\segment\train.modified\masks -n hv

resize-test:
	test -n "$(OUTPUT)"
	test -n "$(INPUT)"
	cd ../pre; python resize.py --encoding jpg -i $(INPUT) -o $(OUTPUT) -r "(756, 1008)" --orientation portrait -t image

resize-train:
	test -n "$(OUTPUT)"
	test -n "$(INPUT)"
	cd ../pre; python resize.py --encoding jpg -i $(INPUT)\\images -o $(OUTPUT)\\images -r "(756, 1008)" --orientation portrait -t image
	cd ../pre; python resize.py --encoding jpg -i $(INPUT)\\masks -o $(OUTPUT)\\masks -r "(756, 1008)" --orientation portrait -t mask

all: data segmented resize
