
weed-detection.bbl: paperpile.bib
	biber weed-detection

weed-detection-sources:
	biber weed-detection

weed-detection: weed-detection.tex weed-detection.bbl
	pdflatex weed-detection.tex

clean:
	rm -f *.synctex*
	rm -f *.aux
	rm -f *.blg
	rm -f *.lof
	rm -f *.log
	rm -f *.run.xml
	rm -f *.gz
