import os.path

import torch
import matplotlib.pyplot as plt
import cv2
import io
from torchvision import transforms
import torchvision.transforms.functional as F
from PIL import Image

from CropDataset import CropDataset
from unet import UNet

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


def single_image_inference(image_pth, model_pth, device):
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

    fig = plt.figure()
    for i in range(1, 3): 
        fig.add_subplot(1, 2, i)
        if i == 1:
            plt.imshow(img, cmap="gray")
        else:
            plt.imshow(pred_mask, cmap="gray")
    plt.show()

    tensor = pred_mask.cpu().numpy()
    tensor = tensor * 255
    cv2.imwrite("mask.png", tensor)
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


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser("Segment image test")
    parser.add_argument("-i", "--image", action="store", required=True, help="Single image or directory of images")
    parser.add_argument("-m", "--model", action="store", required=True, help="Trained model")

    arguments = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if os.path.isdir(arguments.image):
        pred_show_image_grid(arguments.image, arguments.model, device)
    elif os.path.isfile((arguments.image)):
        single_image_inference(arguments.image, arguments.model, device)
    else:
        print(f"Unable to access: {arguments.image}")
        sys.exit(-1)
    sys.exit(0)


    # SINGLE_IMG_PATH = "./data/manual_test/03a857ce842d_15.jpg"
    # DATA_PATH = "./data"
    # MODEL_PATH = "./models/unet.pth"
    #
    # pred_show_image_grid(DATA_PATH, MODEL_PATH, device)
    # single_image_inference(SINGLE_IMG_PATH, MODEL_PATH, device)
