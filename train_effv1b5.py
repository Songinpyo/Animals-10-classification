import torch
import random
import numpy as np
import pandas as pd
from glob import glob
import os
import cv2
import gc
import warnings

from sklearn.model_selection import KFold
from tqdm.auto import tqdm
import torchvision.models as models
from matplotlib import pyplot as plt
from transformers import get_cosine_schedule_with_warmup

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')  # GPU 할당

warnings.filterwarnings('ignore')  # 경고 무시

HYP = {
    'EPOCHS': 100,
    'BATCH_SIZE': 128,
    'LEARNING_RATE': 3e-4,
    'SEED': 41,
    'IMG_SIZE': 256
}


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


seed_everything(HYP['SEED'])

classes = ["butterfly", "cat", "chicken", "cow", "dog", "elephant", "horse", "sheep", "spider", "squirrel"]
train_img = []

for i in classes:
    train_jpg = sorted(glob('train_img/' + f'{i}' + '/' + f'{i}_*.jpg'))
    train_img.extend(train_jpg)


def img_load(path):
    img = cv2.imread(path)[:, :, ::-1]
    img = cv2.resize(img, (HYP['IMG_SIZE'], HYP['IMG_SIZE']))
    gc.collect()
    torch.cuda.empty_cache()
    return img


train_imgs = [img_load(m) for m in tqdm(train_img)]


def files_count(directory_path):
    directory = os.listdir(directory_path)
    return len(directory)


# train data label 부여
train_labels = []
for i in classes:
    length = files_count('train_img/' + f'{i}')
    train_labels.extend([classes.index(f'{i}')] * length)

from albumentations.augmentations.transforms import Normalize
import albumentations as A
from albumentations.pytorch import ToTensorV2

train_transform = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1),
    A.RandomGamma(gamma_limit=(90, 110)),
    A.ShiftScaleRotate(scale_limit=0.1, rotate_limit=10),
    A.Transpose(),
    A.RandomRotate90(),
    A.OneOf([A.NoOp(), A.MultiplicativeNoise(), A.GaussNoise(), A.ISONoise()]),
    A.OneOf(
        [
            A.NoOp(p=0.8),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10),
            A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10)
        ],
        p=0.2,
    ),
    A.OneOf([A.ElasticTransform(), A.GridDistortion(), A.NoOp()]),
    A.Resize(HYP['IMG_SIZE'], HYP['IMG_SIZE']),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # Resnet 참고
    ToTensorV2(),
])

test_transform = A.Compose([
    A.Resize(HYP['IMG_SIZE'], HYP['IMG_SIZE']),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # Resnet 참고
    ToTensorV2()
])

import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score
import time


class Custom_dataset(Dataset):
    def __init__(self, img_paths, labels, mode):
        self.img_paths = img_paths
        self.labels = labels
        self.mode = mode

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = self.img_paths[idx]
        if self.mode == 'train':
            img = train_transform(image=img)

        if self.mode == 'test':
            img = test_transform(image=img)

        label = self.labels[idx]
        return img, label


def count_parameters(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        total_params += param
    print(f"Total Trainable Params: {total_params}")
    return total_params


class efficientnet_b5(nn.Module):
    def __init__(self):
        super(efficientnet_b5, self).__init__()
        self.model = torch.hub.load("pytorch/vision", "efficientnet_b5",
                                    weights="EfficientNet_B5_Weights.IMAGENET1K_V1")

        # num_ftrs = self.model.fc.in_features
        # self.model.fc = nn.Linear(num_ftrs, 10)

        count_parameters(self.model)

    def forward(self, inputs):
        output = self.model(inputs)
        return output


def score_function(real, pred):
    score = f1_score(real, pred, average="macro")
    return score


path = './saved/'
if not os.path.isdir(path):
    os.mkdir(path)


def train(model, optimizer, train_loader, vali_loader, scheduler, device, fold):
    # Loss Function
    criterion = nn.CrossEntropyLoss().to(device)
    scaler = torch.cuda.amp.GradScaler()
    best_f1 = 0

    for epoch in range(1, HYP["EPOCHS"] + 1):
        start = time.time()
        model.train()
        train_loss = 0
        train_pred = []
        train_label = []

        for batch in (train_loader):
            optimizer.zero_grad()
            img = torch.tensor(batch[0]['image'], dtype=torch.float32, device=device)
            label = torch.tensor(batch[1], dtype=torch.long, device=device)

            with torch.cuda.amp.autocast():
                pred = model(img)

            # Calc loss
            loss = criterion(pred, label)

            # backpropagation
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() / len(train_loader)
            train_pred += pred.argmax(1).detach().cpu().numpy().tolist()
            train_label += label.detach().cpu().numpy().tolist()

        train_f1 = score_function(train_label, train_pred)

        TIME = time.time() - start

        epochs = HYP["EPOCHS"]
        print(f'epoch : {epoch}/{epochs}    time : {TIME:.0f}s/{TIME * (epochs - epoch - 1):.0f}s')
        print(f'TRAIN_loss : {train_loss:.5f}  TRAIN_f1 : {train_f1:.5f}')

        if scheduler is not None:
            scheduler.step()

        # Evaluation Validation set
        vali_f1, vali_loss = validation(model, vali_loader, criterion, device)

        print(f'Validation loss : [{vali_loss:.5f}]  Validation f1_score : [{vali_f1:.5f}]')

        # Model Saved
        if best_f1 < vali_f1:
            best_f1 = vali_f1

            torch.save(model.state_dict(), './saved/effv1b5_fold{}.pth'.format(fold))
            print('Model Saved.')


def validation(model, vali_loader, criterion, device):
    model.eval()  # Evaluation

    vali_loss = 0
    vali_pred = []
    vali_label = []

    with torch.no_grad():
        for batch in (vali_loader):
            optimizer.zero_grad()
            img = torch.tensor(batch[0]['image'], dtype=torch.float32, device=device)
            label = torch.tensor(batch[1], dtype=torch.long, device=device)

            # Calc loss
            loss = criterion(pred, label)

            pred = model(img)
            vali_loss += loss.item() / len(vali_loader)

            vali_pred += pred.argmax(1).detach().cpu().numpy().tolist()
            vali_label += label.detach().cpu().numpy().tolist()

        vali_f1 = score_function(vali_label, vali_pred)
    return vali_f1, vali_loss


list_idx = list(np.arange(0, len(train_imgs)))

kf = KFold(n_splits=5, shuffle=True)

for fold, (train_idx, valid_idx) in enumerate(kf.split(list_idx)):
    print("#" * 80)
    print("fold: {}".format(fold))
    train_img_list = [train_imgs[i] for i in train_idx]
    train_label = [train_labels[i] for i in train_idx]

    vali_img_list = [train_imgs[i] for i in valid_idx]
    vali_label = [train_labels[i] for i in valid_idx]

    # Get Dataloader
    train_dataset = Custom_dataset(train_img_list, train_label, mode='train', )
    train_loader = DataLoader(train_dataset, batch_size=HYP['BATCH_SIZE'], shuffle=True, num_workers=16)

    vali_dataset = Custom_dataset(vali_img_list, vali_label, mode='test')
    vali_loader = DataLoader(vali_dataset, batch_size=HYP['BATCH_SIZE'], shuffle=False, num_workers=16)

    model = efficientnet_b5().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=HYP['LEARNING_RATE'])

    total_steps = int(len(train_dataset) * HYP['EPOCHS'] / (HYP['BATCH_SIZE']))
    warmup_ratio = 0.1
    warmup_steps = int(total_steps * warmup_ratio)
    print('total_steps: ', total_steps)
    print('warmup_steps: ', warmup_steps)

    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=total_steps)
    train(model, optimizer, train_loader, vali_loader, scheduler, device, fold)