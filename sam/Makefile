#
# M A K E F I L E
#

default:
	@echo No defalt rule.  Choose one of setup, data

setup:
	test -n "$(SAM_HOME)"
	@echo Install dependancies
	pip install 'git+https://github.com/facebookresearch/segment-anything.git'
	pip install -q supervision
	@echo Install weights
	cd $(SAM_HOME); mkdir -p weights
	cd $(SAM_HOME)/weights; wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth