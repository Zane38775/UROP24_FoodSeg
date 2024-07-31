import os
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
import cv2

class FoodSegDataset(Dataset):
    def __init__(self, data_dir, img_dir, mask_dir, split):  # Remove the default value for 'split'
        self.data_dir = data_dir
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.split = split
        self.img_list = []

        with open(os.path.join(data_dir, 'ImageSets', split + '.txt'), 'r') as f:
            self.img_list = f.readlines()

        self.img_list = [item.replace('\n', '') for item in self.img_list]

        print(f"Found {len(self.img_list)} images in the {split} set")

        transforms = [
            A.Resize(height=256, width=256),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ]

        self.transform = A.Compose(transforms)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.split, self.img_list[idx])
        mask_path = os.path.join(self.mask_dir, self.split, self.img_list[idx].replace('.jpg', '.png'))

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)

        image = cv2.resize(image, (256, 256))
        mask = cv2.resize(mask, (256, 256))

        mask = np.where(mask > 0, 1.0, 0.0)

        augmentations = self.transform(image=image, mask=mask)
        image = augmentations["image"]
        mask = augmentations["mask"]

        return image, mask, self.img_list[idx]