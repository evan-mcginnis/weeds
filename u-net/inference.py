import os.path
from pathlib import Path
import glob

import torch
import torchvision
import matplotlib
# Determine if there is a display
if os.name == 'posix' and "DISPLAY" not in os.environ:
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
import cv2
import io
from torchvision import transforms
import torchvision.transforms.functional as F
from PIL import Image

from CropDataset import CropDataset
from unet import UNet

from tqdm import tqdm

def process(image_pth: str, model_pth, device, output: str, size: tuple):
    model = UNet(in_channels=3, num_classes=1).to(device)
    model.load_state_dict(torch.load(model_pth, map_location=torch.device(device)))

    # If the mask need to be a different size
    if size[0] == 512:
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()])
    else:
        transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor()])


    performance.start()

    img = transform(Image.open(image_pth)).float().to(device)
    originalImage = img
    img = img.unsqueeze(0)


    pred_mask = model(img)

    img = img.squeeze(0).cpu().detach()
    img = img.permute(1, 2, 0)

    pred_mask = pred_mask.squeeze(0).cpu().detach()
    pred_mask = pred_mask.permute(1, 2, 0)
    pred_mask[pred_mask < 0] = 0
    pred_mask[pred_mask > 0] = 1

    performance.stopAndRecord(os.path.basename(image_pth))
    # Name of the image without extension
    image = Path(image_pth).stem

    if output is not None:
        tensor = pred_mask.cpu().numpy()
        tensor = tensor * 255
        outFQN = os.path.join(output, image + "-mask-unet" + "." + arguments.mask_encoding)
        cv2.imwrite(outFQN, tensor)
        outFQN = os.path.join(output, image + "-original" + "." + arguments.image_encoding)
        torchvision.utils.save_image(originalImage, outFQN)
        #cv2.imwrite(outFQN, img.cpu().numpy())
        # img = toImage(pred_mask)
        # img.show()



def pred_show_image_grid(data_path, model_pth, device):
    model = UNet(in_channels=3, num_classes=1).to(device)
    model.load_state_dict(torch.load(model_pth, map_location=torch.device(device)))
    image_dataset = CropDataset(data_path, test=True)
    images = []
    orig_masks = []
    pred_masks = []

    for img, orig_mask in image_dataset:
        img = img.float().to(device)
        img = img.unsqueeze(0)

        pred_mask = model(img)

        img = img.squeeze(0).cpu().detach()
        img = img.permute(1, 2, 0)

        pred_mask = pred_mask.squeeze(0).cpu().detach()
        pred_mask = pred_mask.permute(1, 2, 0)
        pred_mask[pred_mask < 0]=0
        pred_mask[pred_mask > 0]=1

        orig_mask = orig_mask.cpu().detach()
        orig_mask = orig_mask.permute(1, 2, 0)

        images.append(img)
        orig_masks.append(orig_mask)
        pred_masks.append(pred_mask)

        image = transforms.ToPILImage()(pred_mask.unsqueeze(0))
        return_image = io.BytesIO()
        image.save(return_image, "JPEG")

    images.extend(orig_masks)
    images.extend(pred_masks)
    fig = plt.figure()
    for i in range(1, 3*len(image_dataset)+1):
       fig.add_subplot(3, len(image_dataset), i)
       plt.imshow(images[i-1], cmap="gray")
    plt.show()


def single_image_inference(image_pth, model_pth, device, interactive: bool, output: str):
    model = UNet(in_channels=3, num_classes=1).to(device)
    model.load_state_dict(torch.load(model_pth, map_location=torch.device(device)))

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()])

    img = transform(Image.open(image_pth)).float().to(device)
    img = img.unsqueeze(0)
   
    pred_mask = model(img)

    img = img.squeeze(0).cpu().detach()
    img = img.permute(1, 2, 0)

    pred_mask = pred_mask.squeeze(0).cpu().detach()
    pred_mask = pred_mask.permute(1, 2, 0)
    pred_mask[pred_mask < 0]=0
    pred_mask[pred_mask > 0]=1

    # debug
    fig = plt.figure()
    for i in range(1, 3): 
        fig.add_subplot(1, 2, i)
        if i == 1:
            plt.imshow(img, cmap="gray")
        else:
            #image_np = img.permute(1, 2, 0).numpy()
            if pred_mask.shape[2] == 1:
                image_np = pred_mask[:, :, 0]
            plt.imshow(image_np, cmap="gray")

    # Name of the image without extension
    image = Path(image_pth).stem

    if interactive:
        plt.show()
    if output is not None:
        outFQN = os.path.join(output, image +"-plot" + ".png")
        plt.savefig(outFQN)

    plt.close(fig)

    if output is not None:
        tensor = pred_mask.cpu().numpy()
        tensor = tensor * 255
        outFQN = os.path.join(output, image + "-mask" + ".png")
        cv2.imwrite(outFQN, tensor)
        #img = toImage(pred_mask)
        #img.show()

def toImage(tensor):
    # Scale to 0-255 and convert to uint8 if necessary
    tensor_image = (tensor * 255).byte()

    # Convert the tensor to a PIL Image
    pil_image = F.to_pil_image(tensor_image)

    # Save the image
    pil_image.save("mask.png")
    return pil_image

def tuple_type(strings: str):
    strings = strings.replace("(", "").replace(")", "")
    mapped_int = map(int, strings.split(","))
    return tuple(mapped_int)

if __name__ == "__main__":
    import argparse
    import sys
    from Performance import Performance

    parser = argparse.ArgumentParser("Segment image test")
    parser.add_argument("-i", "--image", action="store", required=True, help="Single image or directory of images")
    parser.add_argument("-ie", "--image-encoding", action="store", required=False, default="png", choices=["png", "jpg"], help="Image encoding")
    parser.add_argument("-me", "--mask-encoding", action="store", required=False, default="png", choices=["png", "jpg"], help="Mask encoding")
    parser.add_argument("-m", "--model", action="store", required=True, help="Trained model")
    parser.add_argument("-o", "--output", action="store", required=False, help="Output directory")
    parser.add_argument("-p", "--plot", action="store_true", required=False, default=False, help="Show plot")
    parser.add_argument("-d", "--dimensions", action="store", type=tuple_type, required=False, help="Resize mask to these dimensions (Y, X)")
    parser.add_argument("-c", "--cpu", action="store_true", required=False, default=False, help="Force use of CPU")

    arguments = parser.parse_args()

    performance = Performance("performance.csv")
    (performanceOK, performanceDiagnostics) = performance.initialize()
    if not performanceOK:
        print(performanceDiagnostics)
        sys.exit(1)

    if not arguments.cpu:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = "cpu"
    print(f"Using {device}")

    if not os.path.isfile(arguments.model):
        print(f"Unable to access model: {arguments.model}")
        sys.exit(-1)

    if arguments.output is not None:
        if not os.path.isdir(arguments.output):
            print(f"Unable to access output directory: {arguments.output}")
            sys.exit(-1)

    filesToProcess = []
    if os.path.isdir(arguments.image):
        filesToProcess = glob.glob(os.path.join(arguments.image, "*.jpg"))
        #pred_show_image_grid(arguments.image, arguments.model, device)
    elif os.path.isfile(arguments.image):
        filesToProcess.append(arguments.image)
        #single_image_inference(arguments.image, arguments.model, device)
    else:
        print(f"Unable to access: {arguments.image}")
        sys.exit(-1)
    print(f"Processing {len(filesToProcess)} images")

    for file in tqdm(range(len(filesToProcess))):
        #single_image_inference(file, arguments.model, device, arguments.plot, arguments.output)
        process(filesToProcess[file], arguments.model, device, arguments.output, arguments.dimensions)
    sys.exit(0)


    # SINGLE_IMG_PATH = "./data/manual_test/03a857ce842d_15.jpg"
    # DATA_PATH = "./data"
    # MODEL_PATH = "./models/unet.pth"
    #
    # pred_show_image_grid(DATA_PATH, MODEL_PATH, device)
    # single_image_inference(SINGLE_IMG_PATH, MODEL_PATH, device)
