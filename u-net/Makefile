#
# M A K E F I L E
#
# Hardcoded to check for image-segmentation-keras conda environment
#ENVNAME ?= tf-gpu
#environment:
#ifneq (${CONDA_DEFAULT_ENV},$(ENVNAME))
#      $(error Expected $(ENVNAME) environment to be activated)
#endif

default:
	@echo Targets are data, segmented, and resize
	@echo resize requires INPUT and OUTPUT to be passed in

#
# Copy the results from the remote machine
#
copy:
	scp evan@xavier.weeds.com:~/maricopa-test/segment/output/* /cygdrive/d/maricopa-test/segment/output

# Apply the masks to the images.
# Requites IMAGES, MASKS, and OUTPUT to be supplied

apply-masks:
	test -n "$(IMAGES)"
	test -n "$(MASKS)"
	test -n "$(OUTPUT)"
	cd ../util; python apply-masks.py --image-encoding jpg -i $(IMAGES) --mask-encoding png -m $(MASKS) -o $(OUTPUT)

check: environment
	@echo Good to go

#
# Copy the data that will be used.  This assumes that the masks are already created
#
testit:
	ssh evan@xavier.weeds.com "mkdir -p ~/maricopa-test/segment/originals"

data:
	@echo Copy images
	mkdir -p /cygdrive/d/maricopa-test/segment/originals/masks
	cp -rf /cygdrive/d/maricopa/corrected/2024-11-15/pivot-iphone/masks/* /cygdrive/d/maricopa-test/segment/originals/masks
	cp -rf /cygdrive/d/maricopa/corrected/2024-11-15/flood-iphone/masks/* /cygdrive/d/maricopa-test/segment/originals/masks
	cp -rf /cygdrive/d/maricopa/corrected/2024-11-15/drip-iphone/masks/* /cygdrive/d/maricopa-test/segment/originals/masks
	@echo Copy masks
	mkdir -p /cygdrive/d/maricopa-test/segment/originals/images
	cp  /cygdrive/d/maricopa/corrected/2024-11-15/pivot-iphone/*.jpg /cygdrive/d/maricopa-test/segment/originals/images
	cp  /cygdrive/d/maricopa/corrected/2024-11-15/flood-iphone/*.jpg /cygdrive/d/maricopa-test/segment/originals/images
	cp  /cygdrive/d/maricopa/corrected/2024-11-15/drip-iphone/*.jpg /cygdrive/d/maricopa-test/segment/originals/images
	ssh evan@xavier.weeds.com "mkdir -p ~/maricopa-test/segment/originals"
	rsync  -a /cygdrive/d/maricopa-test/segment/originals/ evan@xavier.weeds.com:maricopa-test/segment/originals/
	@echo Copy test images
	mkdir -p /cygdrive/d/maricopa-test/segment/test.2024-04-24
	mkdir -p /cygdrive/d/maricopa-test/segment/test.2024-04-24.resized
	mkdir -p /cygdrive/d/maricopa-test/segment/test.2024-04-24.final
	cp /cygdrive/d/maricopa/corrected/2024-04-24/iphone-drip/*.jpg  /cygdrive/d/maricopa-test/segment/test.2024-04-24
	cd ../pre; python resize.py -e jpg -i d:/maricopa-test/segment/test.2024-04-24 -o d:/maricopa-test/segment/test.2024-04-24.resized/ -r "(1008, 756)" -t image
	# Reduce to a size evenly divisible by 16
	cd ../pre; python resize.py -e jpg -i d:/maricopa-test/segment/test.2024-04-24 -o d:/maricopa-test/segment/test.2024-04-24.final/ -r "(1008, 752)" -t image
	ssh evan@xavier.weeds.com "mkdir -p ~/maricopa-test/segment/test.2024-04-24.final"
	rsync  -a /cygdrive/d/maricopa-test/segment/test.2024-04-24.final/ evan@xavier.weeds.com:maricopa-test/segment/test.2024-04-24.final/

segmented:
	python ..\lib\VegetationIndex.py -i d:\maricopa-test\segment\originals -o d:\maricopa-test\segment\train.modified\images -m d:\maricopa-test\segment\train.modified\masks -n hv

#
# Images must be resized to dimensions that are evenly divisible by 16
resize-test:
	test -n "$(OUTPUT)"
	test -n "$(INPUT)"
	cd ../pre; python resize.py --encoding jpg -i $(INPUT) -o $(OUTPUT) -r "(756, 1008)" --orientation portrait -t image
	python3 inference.py -m ~/maricopa-test/segment/model/segment20.pth -i ~/maricopa-test/segment/test.2024-04-24.final -d "(752, 1008)" -o ~/maricopa-test/segment/output

resize-train:
	test -n "$(OUTPUT)"
	test -n "$(INPUT)"
	cd ../pre; python resize.py --encoding jpg -i $(INPUT)\\images -o $(OUTPUT)\\images -r "(756, 1008)" --orientation portrait -t image
	cd ../pre; python resize.py --encoding jpg -i $(INPUT)\\masks -o $(OUTPUT)\\masks -r "(756, 1008)" --orientation portrait -t mask

#
# Scale the masks back up so we can evaluate them against the reference masks
# INPUT -- source masks with lower resolutioh
# OUTPUT -- destination for resized masks

resize-for-eval:
	test -n "$(OUTPUT)"
	test -n "$(INPUT)"
	$(eval _OUTPUT = `cygpath -w $(OUTPUT)`)
	$(eval _INPUT = `cygpath -w $(INPUT)`)
	cd ../pre; python resize.py --encoding png -i $(_INPUT) -o $(_OUTPUT) -r "(4032, 3024)"  -t mask

#
# Evaluate the masks produced against the reference
# Resize the images before this target is executed
#
# INPUT -- masks to evaluate
# REFERENCE -- the standard masks
# SOURCE -- images for the masks
# EVALCSV -- Results CSV
#
evaluate:
	test -n "$(INPUT)"
	test -n "$(REFERENCE)"
	test -n "$(SOURCE)"
	test -n "$(CSV)"
	$(eval _INPUT = `cygpath -w $(INPUT)`)
	$(eval _REFERENCE = `cygpath -w $(REFERENCE)`)
	$(eval _SOURCE = `cygpath -w $(SOURCE)`)
	$(eval _CSV = `cygpath -w $(CSV)`)
	cd ../util; python evaluate-masks.py -i $(_INPUT) -t $(_REFERENCE) -s $(_SOURCE) -o $(_CSV) --mask-encoding jpg

# These targets should be executed on the xavier
train:
	test -n "$(MODEL)"
	python3 train.py -d ~/maricopa-test/segment/train.resized/ -b 1 -e 20 -m ~/maricopa-test/segment/model/$(MODEL)

test:
	test -n "$(MODEL)"
	python3 inference.py -m ~/maricopa-test/segment/model/$(MODEL) -i ~/maricopa-test/segment/test.2024-04-24.final -d "(752, 1008)" -o ~/maricopa-test/segment/output

all: data segmented resize
