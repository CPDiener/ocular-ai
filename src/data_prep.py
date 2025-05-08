import os
import shutil

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder


def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Standard ImageNet normalization
                             std=[0.229, 0.224, 0.225])
    ])

def get_dataloaders(data_dir, batch_size=32):
    transform = get_transforms()

    train_ds = ImageFolder(root=f"{data_dir}/train", transform=transform)
    val_ds = ImageFolder(root=f"{data_dir}/val", transform=transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def split_dataset(input_dir, output_dir, val_split=0.2):
    classes = os.listdir(input_dir)
    for cls in classes:
        cls_path = os.path.join(input_dir, cls)
        if not os.path.isdir(cls_path):
            continue
        images = os.listdir(cls_path)
        train_imgs, val_imgs = train_test_split(images, test_size=val_split, random_state=42)
        for split, imgs in zip(["train", "val"], [train_imgs, val_imgs]):
            split_dir = os.path.join(output_dir, split, cls)
            os.makedirs(split_dir, exist_ok=True)
            for img in imgs:
                shutil.copy2(os.path.join(cls_path, img), os.path.join(split_dir, img))


def prepare_dataset():
    input_dir = "../data/raw"
    output_dir = "../data/processed"

    # Check if processed/train and processed/val exist and are non-empty
    already_prepared = (
        os.path.exists(os.path.join(output_dir, "train")) and
        os.path.exists(os.path.join(output_dir, "val")) and
        any(os.scandir(os.path.join(output_dir, "train"))) and
        any(os.scandir(os.path.join(output_dir, "val")))
    )

    if already_prepared:
        print("[INFO] Processed data already exists. Skipping setup.")
    else:
        print("[INFO] Preparing data...")
        split_dataset(input_dir, output_dir, val_split=0.2)
        print("[INFO] Data split complete.")
