
default:
	echo Typo?

clean:
	rm -f *.log

PYTHON = python

segment:
	test -n "$(INPUT)"
	test -n "$(OUTPUT)"
	test -n "$(THRESHOLD)"
	$(eval _INPUT = `cygpath -w $(INPUT)`)
	$(eval _OUTPUT = `cygpath -w $(OUTPUT)`)
	python VegetationIndex.py --input $(_INPUT) --output $(_OUTPUT) --threshold $(THRESHOLD) --masks $(_OUTPUT)
# Show the factors for a type and subtype
factors:
	test -n "$(INPUT)"
	test -n "$(TYPE)"
	test -n "$(SUBTYPE)"
	test -n "$(KIND)"
	$(eval _INPUT = `cygpath -w $(INPUT)`)
	$(PYTHON) Factors.py -i $(_INPUT) -t $(TYPE) -s $(SUBTYPE) -k $(KIND)
