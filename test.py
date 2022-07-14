import torch
import random
import numpy as np
import pandas as pd
from glob import glob
import os
import cv2
import gc
import warnings
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score
import time
from sklearn.model_selection import KFold
from tqdm.auto import tqdm
import torchvision.models as models
from matplotlib import pyplot as plt
from transformers import get_cosine_schedule_with_warmup

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')  # GPU 할당

warnings.filterwarnings('ignore')  # 경고 무시

HYP = {
    'EPOCHS': 50,
    'BATCH_SIZE': 32,
    'LEARNING_RATE': 3e-4,
    'log': True,
    'SEED': 41,
    'IMG_SIZE': 128
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
test_img = []

for i in classes:
    test_jpg = sorted(glob('test_img/' + f'{i}' + '/' + f'{i}_*.jpg'))
    test_img.extend(test_jpg)


def img_load(path):
    img = cv2.imread(path)[:, :, ::-1]
    img = cv2.resize(img, (HYP['IMG_SIZE'], HYP['IMG_SIZE']))
    gc.collect()
    torch.cuda.empty_cache()
    return img

test_imgs = [img_load(n) for n in tqdm(test_img)]

def files_count(directory_path):
    directory = os.listdir(directory_path)
    return len(directory)

# test data label 부여
test_labels = []
for i in classes:
    length = files_count('test_img/' + f'{i}')
    test_labels.extend([classes.index(f'{i}')] * length)

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
        total_params+=param
    print(f"Total Trainable Params: {total_params}")
    return total_params

class efficientnet_v2(nn.Module):
    def __init__(self):
        super(efficientnet_v2, self).__init__()
        self.model = models.efficientnet_v2_s(models.EfficientNet_V2_S_Weights)

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

def predict(model, test_loader, device):
    model.eval()

    test_pred = []
    test_label = []
    correct = 0

    with torch.no_grad():
        for batch in (test_loader):
            img = torch.tensor(batch[0]['image'], dtype=torch.float32, device=device)
            label = torch.tensor(batch[1], dtype=torch.long, device=device)

            # Calc loss
            with torch.cuda.amp.autocast():
                pred = model(img)

            test_pred += pred.argmax(1).detach().cpu().numpy().tolist()
            test_label += label.detach().cpu().numpy().tolist()

            correct += pred.eq(label.view_as(pred)).sum().item()

        test_acc = 100*correct/len(test_labels)
        test_f1 = score_function(test_label, test_pred)
        print(f'test_acc [{test_acc}]', f'test_f1 [{test_f1}]')

    return test_pred

list_idx = list(np.arange(0,len(test_imgs)))

for test_idx in enumerate(list_idx):

    test_img_list = [test_imgs[i] for i in test_idx]
    test_label = [test_labels[i] for i in test_idx]

    test_dataset = Custom_dataset(test_img_list, test_label, mode='test', )
    test_loader = DataLoader(test_dataset, batch_size=HYP["BATCH_SIZE"], shuffle=False, num_workers=0)

checkpoint = torch.load('./saved/effv2_fold{}.pth'.format(4))
model = efficientnet_v2().to(device)
model.load_state_dict(checkpoint)

preds = predict(model, test_loader, device)