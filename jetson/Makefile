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
CONTOURS?=-c
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

# Until the DB is really optional, this is a bit broken
weeds:
	$(PYTHON) weeds.py -i $(INPUT) -o $(OUTPUT) -a $(INDEX) -t $(THRESHOLDS) -df $(TRAINING) $(ALGFLAG) -sc -op $(OPERATION) -lg $(LOG) -d $(DECORATIONS) $(CONTOURS) -ini $(INI) $(STANDALONE)

# This is the one to use
weeds-to-db:
	$(PYTHON) weeds.py -i $(INPUT) -o $(OUTPUT) -a $(INDEX) $(THRESHOLDFLAG) -df $(TRAINING) $(ALGFLAG) -sc -op $(OPERATION) -lg $(LOG) -d $(DECORATIONS) $(CONTOURS) -ini $(INI) $(STANDALONE) -db -alt $(ALTITUDE) -cr $(CROP)

weeds-with-hull:
	$(PYTHON) weeds.py -i $(INPUT) -o $(OUTPUT) -a $(INDEX) $(THRESHOLDFLAG) -df $(TRAINING) $(ALGFLAG) -sc -op $(OPERATION) -lg $(LOG) -d $(DECORATIONS) -ini $(INI) $(STANDALONE) -db -alt $(ALTITUDE) -cr $(CROP) --hull

weeds-plot:
	$(PYTHON) weeds.py -i $(INPUT) -o $(OUTPUT) -a $(INDEX) $(THRESHOLDFLAG) -df $(TRAINING) $(ALGFLAG) -sc -op $(OPERATION) -lg $(LOG) -d $(DECORATIONS) $(CONTOURS) -ini $(INI) $(STANDALONE) -db -alt $(ALTITUDE) -cr $(CROP) -p

report:
	$(PYTHON) analyze.py -df results-current.csv

optimal:
	$(PYTHON) Selection.py -df results-current.csv -fs ALL -f 4 -c 1 -o -p "4-parameters"
	$(PYTHON) Selection.py -df results-current.csv -fs ALL -f 5 -c 1 -o -p "5-parameters"
	$(PYTHON) Selection.py -df results-current.csv -fs ALL -f 6 -c 1 -o -p "6-parameters"
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


