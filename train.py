import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from food_seg import UNET
from data_prep import FoodSegDataset
import time

# Hyperparameters etc.
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
NUM_EPOCHS = 10
NUM_WORKERS = 2
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
PIN_MEMORY = True
LOAD_MODEL = False

DATA_DIR = r'C:\Users\Zane\Documents\UROP24_FoodSeg\data\FoodSeg103'
IMG_DIR = r'C:\Users\Zane\Documents\UROP24_FoodSeg\data\FoodSeg103\Images\img_dir'
MASK_DIR = r'C:\Users\Zane\Documents\UROP24_FoodSeg\data\FoodSeg103\Images\ann_dir'

def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, targets, _) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # 计算准确度
        predictions = (predictions > 0.5).float()
        accuracy = accuracy_score(targets.cpu().numpy().flatten(), predictions.cpu().numpy().flatten())

        # update tqdm loop
        loop.set_postfix(loss=loss.item(), accuracy=accuracy)

def main():
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_dataset = FoodSegDataset(DATA_DIR, IMG_DIR, MASK_DIR, split='train')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        start_time = time.time()
        train_fn(train_loader, model, optimizer, loss_fn, scaler)
        end_time = time.time()
        print(f"Epoch {epoch+1} completed in {end_time-start_time:.2f} seconds")

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        torch.save(checkpoint, f"checkpoint_epoch{epoch+1}.pth.tar")

if __name__ == "__main__":
    main()