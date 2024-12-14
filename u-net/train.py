# T R A I N . P Y
#
# Train a u-net model
#

import argparse
import os.path
import sys

import torch
from torch import optim, nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from unet import UNet
from CropDataset import CropDataset

if __name__ == "__main__":
    LEARNING_RATE = 3e-4
    BATCH_SIZE = 32
    EPOCHS = 2
    # DATA_PATH = "d:\\carvana"
    # MODEL_SAVE_PATH = "d:\\carvana\\model\\unet.pth"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser("Segment image train",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", "--data", action="store", required=True, help="Image and mask directory")
    parser.add_argument("-m", "--model", action="store", required=True, help="Trained model")
    parser.add_argument("-b", "--batch", action="store", required=False, type=int, choices=range(1, 32), metavar="[1-32]", default=BATCH_SIZE, help="Batch size")
    parser.add_argument("-e", "--epochs", action="store", required=False, type=int, choices=range(1, 10), metavar="[1-10]", default=EPOCHS, help="Epochs")

    arguments = parser.parse_args()

    print(f"Train using {device}")

    if not os.path.isdir(arguments.data):
        print(f"Unable to access data directory: {arguments.data}")
        sys.exit(-1)

    # Load the data
    train_dataset = CropDataset(arguments.data)

    # So we get consistent results
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(train_dataset, [0.8, 0.2], generator=generator)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=arguments.batch, shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=arguments.batch, shuffle=True)

    model = UNet(in_channels=3, num_classes=1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    print(f"Training for {arguments.epochs} epochs")
    for epoch in tqdm(range(arguments.epochs)):
        model.train()
        train_running_loss = 0
        for idx, img_mask in enumerate(tqdm(train_dataloader)):
            img = img_mask[0].float().to(device)
            mask = img_mask[1].float().to(device)

            y_pred = model(img)
            optimizer.zero_grad()

            loss = criterion(y_pred, mask)
            train_running_loss += loss.item()
            
            loss.backward()
            optimizer.step()

        train_loss = train_running_loss / (idx + 1)

        model.eval()
        val_running_loss = 0

        with torch.no_grad():
            for idx, img_mask in enumerate(tqdm(val_dataloader)):
                img = img_mask[0].float().to(device)
                mask = img_mask[1].float().to(device)
                
                y_pred = model(img)
                loss = criterion(y_pred, mask)

                val_running_loss += loss.item()

            val_loss = val_running_loss / (idx + 1)

        print("-"*30)
        print(f"Train Loss EPOCH {epoch+1}: {train_loss:.4f}")
        print(f"Valid Loss EPOCH {epoch+1}: {val_loss:.4f}")
        print("-"*30)

    print(f"Saving model")
    torch.save(model.state_dict(), arguments.model)
