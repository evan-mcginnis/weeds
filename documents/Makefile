
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

glossary:
	makeglossaries weed-detection

diffs:
	latexdiff -c latexdiff.cfg weed-detection-old.tex weed-detection.tex > weed-detection-diffs.tex