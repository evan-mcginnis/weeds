univariate:
	python Selection.py -df results.csv -fs univariate -lg debug-selection.yaml
importance:
	python Selection.py -df results.csv -fs importance -lg debug-selection.yaml
pca:
	python Selection.py -df results.csv -fs pca -lg debug-selection.yaml
recursive:
	python Selection.py -df results.csv -fs recursive -lg debug-selection.yaml
