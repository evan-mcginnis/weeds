#
# S E G M E N T  A N Y T H I N G
#
#
import os
import sys
import torch
import argparse
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import cv2
import supervision as sv

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_h"

# Parse command line
parser = argparse.ArgumentParser("Segment Anything")
parser.add_argument("-e", "--encoding", action="store", required=False, default="jpg", choices=["jpg", "png"], help="Image encoding")
parser.add_argument("-i", "--input", action="store", required=True, help="Source image(s)")
parser.add_argument("-o", "--output", action="store", required=True, help="Altered image or directory")
parser.add_argument("-s", "--sam", action="store", required=True, help="SAM Home directory")

arguments = parser.parse_args()

if not os.path.isdir(arguments.sam):
    print(f"Unable to access home: {arguments.sam}")
    sys.exit(-1)

# Create model
CHECKPOINT_PATH = os.path.join(arguments.sam, "weights", "sam_vit_h_4b8939.pth")
if not os.path.isfile(CHECKPOINT_PATH):
    print(f"Unable to access model: {CHECKPOINT_PATH}")
    sys.exit(-1)

sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)

# Mask Generator
mask_generator = SamAutomaticMaskGenerator(sam)

# For now, just process a single image
#imgPath = os.path.join(arguments.sam, "test", arguments.input)
imgPath = arguments.input
image_bgr = cv2.imread(imgPath)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

sam_result = mask_generator.generate(image_rgb)

# Debug -- safe to remove begin
print(sam_result[0].keys())

mask_annotator = sv.MaskAnnotator(color_lookup = sv.ColorLookup.INDEX)

detections = sv.Detections.from_sam(sam_result=sam_result)

annotated_image = mask_annotator.annotate(scene=image_bgr.copy(), detections=detections)

sv.plot_images_grid(
    images=[image_bgr, annotated_image],
    grid_size=(1, 2),
    titles=['source image', 'segmented image']
)
# Debug end

