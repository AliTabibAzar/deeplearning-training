import os
import random
import logging
from collections import Counter

from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split


random.seed(42)

logger = logging.getLogger(__name__)


class UTKFaceDataset(Dataset):
    def __init__(self, names, ages, folder, transform):
        self.names = names
        self.ages = ages
        self.folder = folder
        self.transform = transform

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        path = os.path.join(self.folder, self.names[i])
        img = Image.open(path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        age = torch.tensor(self.ages[i], dtype=torch.float32)

        return img, age


def parse_filename(filename):
    parts = filename.split("_")

    if len(parts) < 3:
        return None

    try:
        age = int(parts[0])
        gender = int(parts[1])
        race = int(parts[2])
    except:
        return None

    if age < 0 or age > 116:
        return None

    return age, gender, race


def get_files(folder):
    names = []
    ages = []
    genders = []
    races = []

    for filename in os.listdir(folder):
        if not filename.lower().endswith(".jpg"):
            continue

        parsed = parse_filename(filename)

        if parsed is None:
            continue

        age, gender, race = parsed

        names.append(filename)
        ages.append(age)
        genders.append(gender)
        races.append(race)

    return names, ages, genders, races


def reduce_kids(names, ages, max_kids=500):
    """Reduce images"""
    kids = []
    others = []

    for i in range(len(ages)):
        if ages[i] <= 3:
            kids.append(i)
        else:
            others.append(i)

    if len(kids) > max_kids:
        kids = random.sample(kids, max_kids)

    keep = kids + others
    random.shuffle(keep)

    names = [names[i] for i in keep]
    ages = [ages[i] for i in keep]

    return names, ages


def get_data(cfg):
    folder = cfg["folder"]
    batch_size = cfg["batch_size"]
    max_kids = cfg.get("max_kids", 500)

    names, ages, genders, races = get_files(folder)

    if len(names) == 0:
        raise FileNotFoundError("No valid images found in dataset folder.")

    logger.info("found %d valid images", len(names))

    male_count = sum(1 for g in genders if g == 0)
    female_count = sum(1 for g in genders if g == 1)
    logger.info("gender count: male=%d, female=%d", male_count, female_count)
    
    race_counts = Counter(races)
    logger.info("race distribution: %s", dict(race_counts))
    names, ages = reduce_kids(names, ages, max_kids)

    logger.info("after balancing: %d images", len(names))

    train_names, val_names, train_ages, val_ages = train_test_split(
        names,
        ages,
        test_size=0.2,
        random_state=42
    )

    normalizer = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalizer
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalizer
    ])

    train_dataset = UTKFaceDataset(train_names, train_ages, folder, train_transform)

    val_dataset = UTKFaceDataset(val_names, val_ages, folder, val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    return train_loader, val_loader