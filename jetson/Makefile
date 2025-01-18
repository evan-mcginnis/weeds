#
# M A K E F I L E
#
clean:
	rm -f *.log*

clean-all:
	rm -f *.log*
	rm -f *.pickle
	#rm -rf ../output/*

requirements:
	pipreqs

#
# The ubuntu command line client is cgps.
#
gps:
	python3 GPSClient.py

grab-arm:
	OPENBLAS_CORETYPE=ARMv8 python3 weeds.py --grab

grab:
	python3 weeds.py -o ../output -lg logging.ini --grab

OS := $(shell uname)

# Begin original Makefile

#
# Executable names
PYTHON = python

# This is for the MAC
ifeq ($(OS),Darwin)
	PYTHON = python3
endif

# Ubuntu
ifeq ($(OS),Linux)
	PYTHON = python3
endif
# This file should exist on a controller.
# If we don't have that bit before the python command, you get a coredump when you import

# This doesn't detect if the platform is a jetson.
# If the nvidia toolkit is installed on a windows system, you see
# this file
#ifeq (,$(wildcard /sys/module/tegra_fuse/parameters/tegra_chip_id))
#	PYTHON = OPENBLAS_CORETYPE=ARMv8 python3
#endif

PYLINT=pylint

#
# Defaults if they are not specified on the make line
#
# Area
AREA?=500
# Log configuration
#LOG?=info-logging.yaml
LOG?=logging.ini
# output directory
OUTPUT?=../../output
# input image set
INPUT?=../../images
# Training data
TRAINING?=training-521.csv
# Decorations on the output images
DECORATIONS?=none
# Machine learning algorithm to use
ML?=lr
# Vegetation index algorithm
INDEX?=ndi
# should be one of: ndi tgi exg exr exgexr cive ngrdi veg com1 mexg com2 rgd
DBHOST?=iron-chef.weeds.com
DBPORT?=27017
DBNAME?=weeds

# Corrections
SUBSET?=train
CORRECTION?=NONE
RATIO?=0:0
ifeq ($(CORRECTION),NONE)
	CORRECTIONFLAGS= -ir $(RATIO)
else
	CORRECTIONFLAGS= --correct  -ir $(RATIO) --imbalance $(CORRECTION) --subset $(SUBSET)
endif

# Proximity to disk
PROXIMITY?=0
ifeq ($(PROXIMITY),0)
	PROXIMITYFLAGS=
else
	PROXIMITYFLAGS=--proximity
endif


# Parameter Selection
# deprecated with move to .INI file
#PARAMETERS?=all-parameters.csv

# Thresholds
# Originally
#THRESHOLDS?="(130,0)"
ifeq ($(THRESHOLDS),otsu)
	THRESHOLDFLAG=
else
	THRESHOLDS?=128
	THRESHOLDS-TGI?="(9,0)"
	THRESHOLDFLAG = -t $(THRESHOLDS)
endif

# Contours
ifeq ($(CONTOURS), "true")
	_CONTOURS=-c
else
	_COUNTOURS=
endif

# Standlone
STANDALONE=--standalone
# Init file
INI?=options.ini

# Logging
LOG=standalone-logging.ini
INI=standalone.ini

# Operation
OPERATION?=normal


# Clean this up a bit, reducing the ML algorithm to one parameter
ALGFLAG = unknown

ifeq ($(ML),knn)
	ALGFLAG = -k
endif
ifeq ($(ML), lr)
	ALGFLAG = -l
endif
ifeq ($(ML), gradient)
	ALGFLAG = -g
endif
ifeq ($(ML), forest)
	ALGFLAG = -f
endif
ifeq ($(ML), decision)
	ALGFLAG = -dt
endif
ifeq ($(ML), svm)
	ALGFLAG = -svm
endif
ifeq ($(ML), lda)
	ALGFLAG = -lda
endif
ifeq ($(ML), mlp)
	ALGFLAG = -mlp
endif


# Crop in image -- not used for all that much
CROP?=unknown
# 0.0 indicates that the image EXIF should be used
ALTITUDE?=0.0

# Indicate that the system should produce treatment plans as an image
TREATMENT?=-sp

LOGFILE = jetson-right.log

# Training
TRAINING?=results-reviewed.csv

OPERATION?=normal

# FACTORS
FACTORS?=4

PREFIX?=reviewed

FORMAT?=PICKLE

TYPE?=all

SUBTYPE?=all

RESULTS?=results

SEGMENTATION?=tgi

SESSION?=unique

#
# W O R K F L O W
#
# 1) Produce the training data with something lik this:
# make weeds-memory TRAINING=d:\maricopa\processed\2024-05-01\pivot-2m\test\training.csv OPERATION=evaluate ML=lda INPUT=c:\uofa\maricopa\corrected\2024-05-01\pivot-2m OUTPUT=d:\maricopa\processed\2024-05-01\pivot-2m THRESHOLD=TRIANGLE INDEX=com2 RESULTS=tmp DECORATIONS=name
# 2) Review all the images and correct the assignments with something like this:
# python review.py -i d:\maricopa\processed\2024-05-01\pivot-2m\test -p processed-*.jpg -o d:\maricopa\processed\2024-05-01\pivot-2m\test\training.csv -a ..\jetson\debug_dataframe.csv --type
# 3) Find the optimal set of parameters to use:
# make optimal TRAINING=d:\maricopa\processed\2024-05-01\pivot-2m\test\training.csv FACTORS=3 PREFIX=maricopa OUTPUT=..\..\output
# 4) Edit the INI file to set the optimal parameters
# 5) Rerun the classification from step 2

# Until the DB is really optional, this is a bit broken
weeds:
	$(PYTHON) weeds.py -i $(INPUT) -o $(OUTPUT) -a $(INDEX) -t $(THRESHOLDS) -df $(TRAINING) $(ALGFLAG) -sc -op $(OPERATION) -lg $(LOG) -d $(DECORATIONS) $(CONTOURS) -ini $(INI) $(STANDALONE)

# This is the one to use
weeds-to-db:
	$(PYTHON) weeds.py -i $(INPUT) -o $(OUTPUT) -a $(INDEX) -df $(TRAINING) $(ALGFLAG) -sc -op $(OPERATION) -lg $(LOG) -d $(DECORATIONS) $(CONTOURS) -ini $(INI) $(STANDALONE) -db -alt $(ALTITUDE) -cr $(CROP) $(CORRECTIONFLAGS) -t $(THRESHOLD) -c

# This assumes cygwin style paths are passed in
# INPUT = path to the images to be processed
# For example
# make weeds-no-db INPUT=/cygdrive/d/maricopa-test/imbalance OUTPUT=/cygdrive/d/maricopa-test/imbalance/processed TRAINING=/cygdrive/d/maricopa-test/imbalance/processed/final/corrected.csv THRESHOLD=TRIANGLE AREA=1200 INDEX=com2 DECORATIONS=name OPERATION=evaluate
weeds-no-db:
	$(eval _INPUT = `cygpath -w $(INPUT)`)
	$(eval _OUTPUT = `cygpath -w $(OUTPUT)`)
	$(eval _TRAINING = `cygpath -w $(TRAINING)`)
	#$(eval _CLASSIFICATION = `cygpath -w $(CLASSIFICATION)`)
	test -n "$(_TRAINING)"
	#$(PYTHON) weeds.py -i $(_INPUT) -o $(_OUTPUT) -a $(INDEX) -df $(_TRAINING) $(ALGFLAG) -sc -op $(OPERATION) -lg $(LOG) -d $(DECORATIONS) $(CONTOURS) -ini $(INI) $(STANDALONE)  -alt $(ALTITUDE) -cr $(CROP) $(CORRECTIONFLAGS) -t $(THRESHOLD) -c -ma $(AREA) --session-name $(SESSION)
	$(PYTHON) weeds.py -i $(_INPUT) -o $(_OUTPUT) -a $(INDEX) -df $(_TRAINING) $(ALGFLAG) -sc -op $(OPERATION) -lg $(LOG) -d $(DECORATIONS)  $(_CONTOURS) -ini $(INI) $(STANDALONE)  -alt $(ALTITUDE) -cr $(CROP) $(CORRECTIONFLAGS) -t $(THRESHOLD) -ma $(AREA) --session-name $(SESSION)

# Debug memory leak
weeds-memory:
	$(PYTHON) weeds.py -i $(INPUT) -o $(OUTPUT) -a $(INDEX) -df $(TRAINING) $(ALGFLAG) -sc -op $(OPERATION) -lg $(LOG) -d $(DECORATIONS) -ini $(INI) $(STANDALONE) -alt $(ALTITUDE) -cr $(CROP) $(CORRECTIONFLAGS) -t $(THRESHOLD) -r $(RESULTS) $(PROXIMITYFLAGS)

weeds-with-hull:
	$(PYTHON) weeds.py -i $(INPUT) -o $(OUTPUT) -a $(INDEX) $(THRESHOLDFLAG) -df $(TRAINING) $(ALGFLAG) -sc -op $(OPERATION) -lg $(LOG) -d $(DECORATIONS) -ini $(INI) $(STANDALONE) -db -alt $(ALTITUDE) -cr $(CROP) --hull

weeds-plot:
	$(PYTHON) weeds.py -i $(INPUT) -o $(OUTPUT) -a $(INDEX) -df $(TRAINING) $(ALGFLAG) -sc -op $(OPERATION) -lg $(LOG) -d $(DECORATIONS) $(CONTOURS) -ini $(INI) $(STANDALONE) -db -alt $(ALTITUDE) -cr $(CROP) -p -t $(THRESHOLD)

comprehensive-plot:
	$(PYTHON) weeds.py  -i $(INPUT) -o d:\comp-out -a $(SEGMENTATION) -df $(TRAINING) --knn -sc -op normal -lg standalone-logging.ini -d name -c -ini comprehensive.ini --standalone -db -alt 0.0 -cr unknown -r lettuce -t $(THRESHOLD) -p --minarea 750

comprehensive:
	$(PYTHON) weeds.py  -i $(INPUT) -o d:\comp-out -a $(SEGMENTATION) -df $(TRAINING) $(ALGFLAG) -sc -op $(OPERATION) -lg standalone-logging.ini -d name -c -ini comprehensive.ini --standalone -db -alt 0.0 -cr unknown -r lettuce -t $(THRESHOLD) --minarea 750
	#$(PYTHON) weeds.py  -i d:\comprehensive -o d:\comp-out -a tgi  -df d:\comprehensive\lettuce-corrected.csv --knn -sc -op evaluate -lg standalone-logging.ini -d name -c -ini comprehensive.ini --standalone -db -alt 0.0 -cr unknown -r lettuce -t 6

comprehensive-test:
	$(PYTHON) weeds.py  -i $(INPUT) -o d:\comp-test -a $(SEGMENTATION) -df $(TRAINING) $(ALGFLAG) -sc -op normal -lg standalone-logging.ini -d name -c -ini comprehensive.ini --standalone -db -alt 0.0 -cr unknown -r lettuce -t $(THRESHOLD) --minarea 750


# This will produce the table that can be copied into a document
FORMAT?=latex
LOGGING?=standalone-logging.ini
optimal-results:
	$(eval _PARAMETERS = `cygpath -w $(PARAMETERS)`)
	$(eval _RESULTS = `cygpath -w $(RESULTS)`)
	$(eval _ALL_RESULTS = `cygpath -w $(ALL_RESULTS)`)
	test -n "$(TYPE)"
	test -n "$(SUBTYPE)"
	test -n "$(FORMAT)"
	test -n "$(LOGGING)"
	test -n "$(CAPTION_LONG)"
	test -n "$(CAPTION_SHORT)"
	test -n "$(LABEL)"
	test -n "$(PREFIX)"
	#$(PYTHON) results.py -r $(_RESULTS) -c $(_PARAMETERS) -o $(FORMAT) --type $(TYPE) --subtype $(SUBTYPE) -lg $(LOGGING) --directory $(_ALL_RESULTS) -n 2 --long "$(CAPTION_LONG)"  --short "$(CAPTION_SHORT)" --label $(LABEL)
	$(PYTHON) results.py --prefix $(PREFIX) -c $(_PARAMETERS) -o $(FORMAT) --type $(TYPE) --subtype $(SUBTYPE) -lg $(LOGGING) --directory $(_ALL_RESULTS) --long "$(CAPTION_LONG)"  --short "$(CAPTION_SHORT)" --label $(LABEL) --summary

PREFIX?=comprehensive
OUTPUT?=c:\comp-results
TYPES?=TEXTURE
comprehensive-optimal:
	$(PYTHON) Selection.py -df $(TRAINING) -ty $(TYPES) -fs ALL -f 4 -l -of CSV -lg logging.ini -p $(PREFIX) -o -od $(OUTPUT)

report:
	$(PYTHON) analyze.py -df results-current.csv

# Report the selections to a file that can then be subsequently read
# This needs to be done before the results are reported. An example:
# make selections TRAINING=d:\comprehensive\mexg-manual-corrected.csv FACTORS=4 PREFIX=comprehensive TYPE=TEXTURE SUBTYPE=all FORMAT=PICKLE PREFIX=comprehensive
selections:
	$(PYTHON) Selection.py -df $(TRAINING) -fs ALL -f $(FACTORS) -l -lg logging.ini -of $(FORMAT) -co -ty $(TYPE) -s $(SUBTYPE) -p $(PREFIX)

analyze:
	$(eval _INPUT = `cygpath -w $(INPUT)`)
	cd ../lib;$(PYTHON) Manipulate.py -i $(_INPUT)

# Iterate over all the factors taken N at a time to find the optimum
# OUTPUT -- location of output files
# TYPE -- color, shape, texture, or ALL
# SUBTYPE -- LBP, GLCM, HOG, or ALL
# FACTORS -- number of factors
# PREFIX -- text to precede the output, like "dissertation"
optimal:
	test -n "$(FACTORS)"
	test -n "$(TYPE)"
	test -n "$(SUBTYPE)"
	test -n "$(PREFIX)"
	$(eval _TRAINING = `cygpath -w $(TRAINING)`)
	$(eval _OUTPUT = `cygpath -w $(OUTPUT)`)
	$(PYTHON) Selection.py -df $(_TRAINING) -fs ALL -f $(FACTORS) -l -lg standalone-logging.ini -p $(PREFIX) -o -ty $(TYPE) -s $(SUBTYPE) -od $(_OUTPUT)
	cp jetson.log $(_OUTPUT)

# Test the various types and subtypes -- find the optimal set and then write that out
optimal-detailed:
	$(eval _TRAINING = `cygpath -w $(TRAINING)`)
	$(PYTHON) Selection.py -df $(TRAINING) -fs ALL -f $(FACTORS) -l -lg unattended.ini -p COLOR -o -ty COLOR
	$(PYTHON) Selection.py -df $(TRAINING) -fs ALL -f $(FACTORS) -l -lg unattended.ini -of $(FORMAT) -ty COLOR
	mv parameters.pickle COLOR.parameters.pickle
	$(PYTHON) Selection.py -df $(TRAINING) -fs ALL -f $(FACTORS) -l -lg unattended.ini -p TEXTURE -o -ty TEXTURE
	$(PYTHON) Selection.py -df $(TRAINING) -fs ALL -f $(FACTORS) -l -lg unattended.ini -of $(FORMAT) -ty TEXTURE
	mv parameters.pickle TEXTURE.parameters.pickle
	$(PYTHON) Selection.py -df $(TRAINING) -fs ALL -f 3 -l -lg unattended.ini -p HOG -o -ty TEXTURE -s HOG
	$(PYTHON) Selection.py -df $(TRAINING) -fs ALL -f 3 -l -lg unattended.ini -of $(FORMAT) -ty TEXTURE -s HOG
	mv parameters.pickle HOG.parameters.pickle
	$(PYTHON) Selection.py -df $(TRAINING) -fs ALL -f 3 -l -lg unattended.ini -p LBP -o -ty TEXTURE -s LBP
	$(PYTHON) Selection.py -df $(TRAINING) -fs ALL -f 3 -l -lg unattended.ini -of $(FORMAT) -ty TEXTURE -s LBP
	mv parameters.pickle LBP.parameters.pickle
	$(PYTHON) Selection.py -df $(TRAINING) -fs ALL -f $(FACTORS) -l -lg unattended.ini -p GLCM -o -ty TEXTURE -s GLCM
	$(PYTHON) Selection.py -df $(TRAINING) -fs ALL -f $(FACTORS) -l -lg unattended.ini -of $(FORMAT) -ty TEXTURE -s GLCM
	mv parameters.pickle GLCM.parameters.pickle
	$(PYTHON) Selection.py -df $(TRAINING) -fs ALL -f $(FACTORS) -l -lg unattended.ini -p SHAPE -o -ty SHAPE
	$(PYTHON) Selection.py -df $(TRAINING) -fs ALL -f $(FACTORS) -l -lg unattended.ini -of $(FORMAT) -ty SHAPE
	mv parameters.pickle SHAPE.parameters.pickle

svm:
	$(PYTHON) weeds.py -r ../../results/results.svm.csv -i $(INPUT) -o ../../results/output-svm -a $(INDEX) -t $(THRESHOLDS) -df $(TRAINING) -svm -sc -v -lg $(LOG) -se $(PARAMETERS) -d $(DECORATIONS) $(CONTOURS) -ini $(INI) $(STANDALONE)
	mv $(LOGFILE) ../../results/output-svm.log

tree:
	$(PYTHON) weeds.py -r ../../results/results.tree.csv -i $(INPUT) -o ../../results/output-tree -a $(INDEX) -t $(THRESHOLDS) -df $(TRAINING) -dt -sc -v -lg $(LOG) -se $(PARAMETERS) -d $(DECORATIONS) $(CONTOURS) -ini $(INI) $(STANDALONE)
	mv $(LOGFILE) ../../results/output-tree.log

knn:
	$(PYTHON) weeds.py -r ../../results/results.knn.csv -i $(INPUT) -o ../../results/output-knn -a $(INDEX) -t $(THRESHOLDS) -df $(TRAINING) -k -sc -v -lg $(LOG) -se $(PARAMETERS) -d $(DECORATIONS) $(CONTOURS) -ini $(INI) $(STANDALONE)
	mv $(LOGFILE) ../../results/output-knn.log

gradient:
	$(PYTHON) weeds.py -r ../../results/results.gradient.csv -i $(INPUT) -o ../../results/output-gradient -a $(INDEX) -t $(THRESHOLDS) -df $(TRAINING) -g -sc -v -lg $(LOG) -se $(PARAMETERS) -d $(DECORATIONS) $(CONTOURS) -ini $(INI) $(STANDALONE)
	mv $(LOGFILE) ../../results/output-gradient.log

lr:
	$(PYTHON) weeds.py -r ../../results/results.lr.csv -i $(INPUT) -o ../../results/output-lr -a $(INDEX) -t $(THRESHOLDS) -df $(TRAINING) -l -sc -v -lg $(LOG) -se $(PARAMETERS) -d $(DECORATIONS) $(CONTOURS) -ini $(INI) $(STANDALONE)
	mv $(LOGFILE) ../../results/output-lr.log

forest:
	$(PYTHON) weeds.py -r ../../results/results.forest.csv -i $(INPUT) -o ../../results/output-forest -a $(INDEX) -t $(THRESHOLDS) -df $(TRAINING) -f -sc -v -lg $(LOG) -se $(PARAMETERS) -d $(DECORATIONS) $(CONTOURS) -ini $(INI) $(STANDALONE)
	mv $(LOGFILE) ../../results/output-forest.log

all-techniques: svm tree knn gradient lr forest
	echo "Done"



weeds-extract:
	$(PYTHON) weeds.py -x -i $(INPUT) -o $(OUTPUT) -a $(INDEX) -t $(THRESHOLDS) -df $(TRAINING) $(ALGFLAG) -sc -v -lg $(LOG) -se $(PARAMETERS) -d $(DECORATIONS) $(CONTOURS) -ini $(INI) $(STANDALONE)

#weeds-plot:
#	$(PYTHON) weeds.py -i $(INPUT) -o $(OUTPUT) -a $(INDEX) -t $(THRESHOLDS-TGI) -df $(TRAINING) $(ALGFLAG) -sc -v -lg $(LOG) -se $(PARAMETERS) -d $(DECORATIONS) $(CONTOURS) -ini $(INI) $(STANDALONE) --plot

weeds-glcm:
	$(PYTHON) weeds.py -i $(INPUT) -o $(OUTPUT) -a $(INDEX) -t $(THRESHOLDS-TGI) -df $(TRAINING) $(ALGFLAG) -sc -v -lg $(LOG) -se $(PARAMETERS) -d $(DECORATIONS) $(CONTOURS) -ini $(INI) $(STANDALONE)

weeds-debug:
	$(PYTHON) weeds.py -i $(INPUT) -o $(OUTPUT) -a $(INDEX) -t $(THRESHOLDS-TGI) -df $(TRAINING) $(ALGFLAG) -sc -v -lg $(LOG) -se $(PARAMETERS) -d $(DECORATIONS) $(CONTOURS) -ini $(INI) $(STANDALONE) -ma $(AREA) -x

treatment:
	$(PYTHON) weeds.py -i $(INPUT) -o $(OUTPUT) -a $(INDEX) -t $(THRESHOLDS) -df $(TRAINING) $(ALGFLAG) -sc -v -lg $(LOG) -se $(PARAMETERS) -d location $(TREATMENT) -ini $(INI) $(STANDALONE)

lint:
	$(PYLINT) weeds.py

# H P C C
DATA?=normalized.csv

hpcc-upload:
	scp $(DATA) evanmc@filexfer.hpc.arizona.edu:

# This should be run only on an ubuntu distribution
OPENSSL = openssl
GENRSA = genrsa
SERVERKEY = server.key
PASS = weeds
CSR = server.csr
CERT = server.crt

# Certificate authority stuff
SSLHOME = /etc/ssl
CA = CA
CAHOME = $(SSLHOME)/$(CA)
CAKEYS = $(CAHOME)/private
CACERTS = $(CAHOME)
NEWCERTS = newcerts
NEWCERTSHOME = $(SSLHOME)/$(CA)/$(NEWCERTS)
CASERIAL = $(CAHOME)/serial
CAINDEX = $(CAHOME)/index.txt
CAROOTCERT = cacert.pem
CAROOTKEY = cakey.pem

# Tested with this in openssl.cnf
#[ CA_default ]
#
#dir             = ./CA                  # Where everything is kept
#certs           = $dir/certs            # Where the issued certs are kept
#crl_dir         = $dir/crl              # Where the issued crl are kept
#database        = $dir/index.txt        # database index file.
##unique_subject = no                    # Set to 'no' to allow creation of
#                                        # several certs with same subject.
#new_certs_dir   = $dir/newcerts         # default place for new certs.
#
#certificate     = $dir/cacert.pem       # The CA certificate
#serial          = $dir/serial           # The current serial number
#crlnumber       = $dir/crlnumber        # the current crl number
#                                        # must be commented out to leave a V1 CRL
#crl             = $dir/crl.pem          # The current CRL
#private_key     = $dir/private/cakey.pem# The private key



$(CAHOME):
	mkdir $(CAHOME)

$(NEWCERTSHOME):
	mkdir $(NEWCERTSHOME)

$(CAKEYS):
	mkdir $(CAKEYS)

#$(CACERTS):
#	mkdir $(CACERTS)

$(CASERIAL): $(CAHOME)
	sh -c "echo '01' > /etc/ssl/CA/serial"

$(CAINDEX): $(CAHOME)
	touch $(CAINDEX)

$(CAKEYS)/$(CAROOTKEY):
	$(OPENSSL) req -new -x509 -extensions v3_ca -keyout $(CAROOTKEY) -out $(CAROOTCERT) -days 3650 -subj "/C=US/ST=Arizona/L=Tucson/O=University of Arizona/CN=FoobarCA"
	mv $(CAROOTKEY) $(CAKEYS)
	mv $(CAROOTCERT) $(CACERTS)

CA: $(CAHOME) $(NEWCERTSHOME) $(CASERIAL) $(CAINDEX) $(CACERTS) $(CAKEYS) $(CAKEYS)/$(CAROOTKEY)
	echo "Certificate authority is " $(CAHOME)

# Certificate items

$(SERVERKEY):
	$(OPENSSL) genrsa -des3 -out $(SERVERKEY) -passout pass:$(PASS) 2048
	$(OPENSSL) rsa -check -in $(SERVERKEY) -passin pass:$(PASS)

$(SERVERKEY).insecure: $(SERVERKEY)
	$(OPENSSL) rsa -in $(SERVERKEY) -out $(SERVERKEY).insecure
	mv $(SERVERKEY) $(SERVERKEY).secure
	mv $(SERVERKEY).insecure $(SERVERKEY)

$(CSR):
	$(OPENSSL) req -new -key $(SERVERKEY) -out $(CSR)

$(CERT): $(CSR)
	$(OPENSSL) x509 -req -days 365 -in $(CSR) -signkey $(SERVERKEY) -out $(CERT)

certs: $(SERVERKEY) $(SERVERKEY).insecure $(CSR) $(CERT)
	echo "Certificates complete. "

signed:
	cd $(SSLHOME); $(OPENSSL) ca -in $(CSR) -config $(SSLHOME)/openssl.cnf
	echo "Install this with 1) copy crt to /usr/local/share/ca-cerficates 2) update-ca-certificates"

hsv:
	$(PYTHON) view-hsv.py -i $(OUTPUT)/original-11.jpg

test-capture:
	$(PYTHON) CameraFile.py -s sample.jpg -o options.ini -a -l debug-logging.yaml

# L A T E X
thesis:
	pdflatex thesis.tex

# Various targets that are probably not needed anymore

#heuristic:
#	$(PYTHON) weeds.py -i $(INPUT) -o $(OUTPUT) -a ndi -t "(130,0)" -v
#
#development:
#	$(PYTHON) weeds.py -i $(INPUT) -o $(OUTPUT) -a ndi -t "(130,0)" -df $(TRAINING) -l -sc -v -d $(DECORATIONS)
#
#minimal:
#	$(PYTHON) weeds.py -i $(INPUT) -o $(OUTPUT) -a ndi -t "(130,0)" -df $(TRAINING) -l -sc -v -d none
#
#
#lr-hsv:
#	$(PYTHON) weeds.py -i $(INPUT)4 -o $(OUTPUT) -a ndi -t "(130,0)" -df $(TRAINING) -l -sc -v
#
#lr:
#	$(PYTHON) weeds.py -i $(INPUT) -o $(OUTPUT) -a ndi -t "(130,0)" -df $(TRAINING) -l -sc -v
#
#lr3:
#	$(PYTHON) weeds.py -i $(INPUT)3 -o $(OUTPUT) -a ndi -t "(130,0)" -df $(TRAINING) -l -sc -v
#
#knn:
#	$(PYTHON) weeds.py -i $(INPUT) -o $(OUTPUT) -a ndi -t "(130,0)" -df $(TRAINING) -k -sc -v
#
#decision:
#	$(PYTHON) weeds.py -i $(INPUT) -o $(OUTPUT) -a ndi -t "(130,0)" -df $(TRAINING) -dt -sc -v
#
#forest:
#	$(PYTHON) weeds.py -i $(INPUT) -o $(OUTPUT) -a ndi -t "(130,0)" -df $(TRAINING) -f -sc -v
#
#gradient:
#	$(PYTHON) weeds.py -i $(INPUT) -o $(OUTPUT) -a ndi -t "(130,0)" -df $(TRAINING) -g -sc -v


treatment2:
	$(PYTHON) weeds.py -i $(INPUT) -o $(OUTPUT) -a ndi -t "(130,0)" -df training-doubled.csv -l -sc -v -sp -d location

crop-images:
	$(PYTHON) weeds.py -i $(INPUT) -o $(OUTPUT) -a ndi -t "(130,0)" -df $(TRAINING) -k

proposal.bbl: paperpile.bib
	biber proposal

proposal-sources:
	biber proposal

proposal: proposal.tex proposal.bbl
	pdflatex proposal.tex


