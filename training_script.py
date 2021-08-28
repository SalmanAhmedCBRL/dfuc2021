

########################
#Default backbone is B0, but can be changed to any backbone
baseline_name = "tf_efficientnet_b0"

########################



import os
import gc
import copy
import random
import shutil
import typing as tp
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from torch.optim.lr_scheduler import CosineAnnealingLR
from warmup_scheduler import GradualWarmupScheduler
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torch.cuda import amp
import timm
import albumentations as A
import albumentations
from albumentations.pytorch import ToTensorV2
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import cv2
import time
from pylab import rcParams
rcParams['figure.figsize'] = 20,10
import warnings
warnings.simplefilter('ignore')
import gc
import torchvision
gc.enable()

# %% [code]
def set_seed(seed = 0):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    np.random.seed(seed)
    random_state = np.random.RandomState(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    return random_state


RANDAM_SEED = 42
random_state = set_seed(RANDAM_SEED)

# %% [code]
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")

# %% [code]
transforms_train = albumentations.Compose([
    albumentations.HorizontalFlip(p=0.5),
    albumentations.RandomBrightness(limit=0.2, p=0.75),
    albumentations.RandomContrast(limit=0.2, p=0.75),
    
    albumentations.OneOf([
        albumentations.Blur(p=0.75),
        albumentations.MedianBlur(blur_limit=5, p=0.75),
        albumentations.GaussianBlur(p=0.75),
        albumentations.MotionBlur(p=0.75)
    ], p=0.75),

    albumentations.OneOf([
        albumentations.OpticalDistortion(distort_limit=1.),
        albumentations.GridDistortion(num_steps=5, distort_limit=1.),
    ], p=0.75),
 
    albumentations.HueSaturationValue(hue_shift_limit=40, sat_shift_limit=40, val_shift_limit=0, p=0.75),
    albumentations.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.3, rotate_limit=30, border_mode=0, p=0.75),
])

# %% [code]
train = pd.read_csv("/data0/Salman/alcer/DFUC2021_train/train.csv")
train = train[0:5955]
train["image"] = "/data0/Salman/alcer/DFUC2021_train/images/" + train["image"]
train.head()

# %% [code]
print (500 * 0.428, 500 * 0.454, 500 * 0.03, 500 * 0.088)
new_y = [0] * 214
new_y.extend([1] * 227)
new_y.extend([2] * 15)
new_y.extend([3] * 44)

# %% [code]
2552 / 5955, 2555 / 5955, 227 / 5955, 621 / 5955

# %% [code]
class_weights_c = compute_class_weight(class_weight='balanced', classes=[0, 1, 2, 3], y=new_y)
class_weights_c

# %% [code]
y = np.argmax(train[['none', 'infection', 'ischaemia', 'both']].values, axis=1)
y = y.astype('uint8')
np.unique(y, return_counts=True)

# %% [code]
N_FOLDS = 5
n = 4
n_workers = 16
device = "cuda:2"
Batch_Size = 64
warmup_epo = 1
init_lr = 1e-4
cosine_epo = 89
n_epochs = warmup_epo + cosine_epo

# %% [code]
class_weights_c = torch.tensor(class_weights_c)
class_weights_c = class_weights_c.to(device, dtype=torch.float)

# %% [code]
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDAM_SEED)
train["fold"] = -1
for fold_id, (_, val_idx) in enumerate(skf.split(train["image"], y)):
    train.loc[val_idx, "fold"] = fold_id

# %% [code]
IMG = train["image"].values
T1 = train["none"].values
T2 = train["infection"].values
T3 = train["ischaemia"].values
T4 = train["both"].values
FOLDS = train["fold"].values
train.head()

# %% [code]
def get_f1(valid_targets, predictions):
    scores = []
    k = 0
    while k < n:
        c = valid_targets == k
        sc = f1_score(c, np.round(predictions[:, k]), average='macro')
        scores.append(sc)
        k += 1
    ov = f1_score(valid_targets, np.argmax(predictions, axis=1), average='macro')
    return scores, ov

# %% [code]
class GradualWarmupSchedulerV2(GradualWarmupScheduler):
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        super(GradualWarmupSchedulerV2, self).__init__(optimizer, multiplier, total_epoch, after_scheduler)
    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

# %% [code]
class ClassificationDataset:
    
    def __init__(self, image_paths, targets0, targets1, targets2, targets3, transform): 
        self.image_paths = image_paths
        self.targets0 = targets0
        self.targets1 = targets1
        self.targets2 = targets2
        self.targets3 = targets3
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, item):      
        
        image = cv2.imread(self.image_paths[item])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            image = self.transform(image = image)["image"]
        
        image = image.transpose((2, 0, 1)) / 255.0
        
        
        targets0 = self.targets0[item]
        targets1 = self.targets1[item]
        targets2 = self.targets2[item]
        targets3 = self.targets3[item]
        all_targets = np.argmax([targets0, targets1, targets2, targets3])
                
        return {
            "image": torch.tensor(image, dtype=torch.float),
            "targets0": torch.tensor(all_targets, dtype=torch.float),
        }

# %% [code]
class CustomModel(nn.Module):
    def __init__(self, model_name, pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=n)

    def forward(self, x):
        output = self.model(x)
        return output

# %% [code]
def train_one_epoch(data_loader, model, optimizer, device):
    
    criterion = nn.CrossEntropyLoss(weight = class_weights_c)
    model.train()
    
    scaler = amp.GradScaler(enabled=True)
    for data in tqdm(data_loader, position=0, leave=True, desc='Training'):
        
        inputs = data["image"]
        targets0 = data['targets0']
        

        inputs = inputs.to(device, dtype=torch.float)
        targets0 = targets0.to(device, dtype=torch.long)
        
        optimizer.zero_grad()
        
        with amp.autocast(True):
            outputs = model(inputs)
            loss = criterion(outputs, targets0)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
def evaluate(data_loader, model, device):
    model.eval()
    
    final_targets = []
    final_outputs = []
    act = torch.nn.Softmax(dim=1)
    with torch.no_grad():
        
        for data in tqdm(data_loader, position=0, leave=True, desc='Evaluating'):
            inputs = data["image"]
            targets = data["targets0"]
            
            inputs = inputs.to(device, dtype=torch.float)
            targets = targets.to(device, dtype=torch.long)
            
            output = model(inputs)
            output = act(output)
            
            targets = targets.detach().cpu().numpy().tolist()
            output = output.detach().cpu().numpy().tolist()
            
            final_targets.extend(targets)
            final_outputs.extend(output)
            
    return final_outputs, final_targets

# %% [code]
m = CustomModel(model_name=baseline_name, pretrained=True)
optimizer = optim.Adam(m.parameters(), lr=init_lr)
scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cosine_epo)
scheduler_warmup = GradualWarmupSchedulerV2(optimizer, multiplier=10, total_epoch=warmup_epo, after_scheduler=scheduler_cosine)
lrs = []
for epoch in range(1, n_epochs+1):
    scheduler_warmup.step(epoch-1)
    lrs.append(optimizer.param_groups[0]["lr"])
rcParams['figure.figsize'] = 20,3
del m
gc.collect()
plt.plot(lrs)

# %% [code]
for fold in range(0, N_FOLDS):
    
    test_index = train.iloc[FOLDS == fold].index.tolist()
    train_index = train.iloc[FOLDS != fold].index.tolist()

    train_images, valid_images = IMG[train_index], IMG[test_index]
    train_targets0, valid_targets0 = T1[train_index], T1[test_index]
    train_targets1, valid_targets1 = T2[train_index], T2[test_index]
    train_targets2, valid_targets2 = T3[train_index], T3[test_index]
    train_targets3, valid_targets3 = T4[train_index], T4[test_index]
    train_dataset = ClassificationDataset(train_images, train_targets0, 
                                          train_targets1, train_targets2, train_targets3, transforms_train)
    valid_dataset = ClassificationDataset(valid_images, valid_targets0, valid_targets1, valid_targets2, valid_targets3, None)
    xtrain_dataset = ClassificationDataset(train_images, train_targets0, 
                                          train_targets1, train_targets2, train_targets3, None)
    xtrain_loader = torch.utils.data.DataLoader(xtrain_dataset, batch_size=Batch_Size,shuffle=True, num_workers=n_workers)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=Batch_Size,shuffle=True, num_workers=n_workers)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=Batch_Size,shuffle=False, num_workers=n_workers)
    break

# %% [code]
for each in xtrain_loader:
    image = each["image"]
    labels = each["targets0"].to(dtype=torch.long)
    grid = torchvision.utils.make_grid(image)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    plt.axis('off')
    plt.title(labels.numpy());
    break

# %% [code]
for each in train_loader:
    image = each["image"]
    labels = each["targets0"].to(dtype=torch.long)
    grid = torchvision.utils.make_grid(image)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    plt.axis('off')
    plt.title(labels.numpy());
    break

# %% [code]
for each in valid_loader:
    image = each["image"]
    labels = each["targets0"].to(dtype=torch.long)
    grid = torchvision.utils.make_grid(image)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    plt.axis('off')
    plt.title(labels.numpy());
    break

# %% [code]


# %% [code] {"scrolled":true}
for fold in range(0, N_FOLDS):
    
    log_file = "/data0/Salman/alcer/models/" + baseline_name + "_" + str(fold) + "_log.txt"

    content = 'Fold: ' + str(fold)
    print(content)
    with open(log_file, 'a') as appender:
        appender.write(content + '\n')    

    model = CustomModel(model_name=baseline_name, pretrained=True)
    model.to(device)

    test_index = train.iloc[FOLDS == fold].index.tolist()
    train_index = train.iloc[FOLDS != fold].index.tolist()

    train_images, valid_images = IMG[train_index], IMG[test_index]
    train_targets0, valid_targets0 = T1[train_index], T1[test_index]
    train_targets1, valid_targets1 = T2[train_index], T2[test_index]
    train_targets2, valid_targets2 = T3[train_index], T3[test_index]
    train_targets3, valid_targets3 = T4[train_index], T4[test_index]

    train_dataset = ClassificationDataset(train_images, train_targets0, 
                                          train_targets1, train_targets2, train_targets3, transforms_train)
    valid_dataset = ClassificationDataset(valid_images, valid_targets0, valid_targets1, valid_targets2, valid_targets3, None)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=Batch_Size,shuffle=True, num_workers=n_workers)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=Batch_Size,shuffle=False, num_workers=n_workers)
    valid_targets = np.argmax(np.column_stack([valid_targets0, valid_targets1, valid_targets2, valid_targets3]), axis=1)
    un, co = np.unique(valid_targets, return_counts=True)
    co = co / valid_targets.shape[0]
    content = "Validation Distribution 0, 1, 2, 3 -> " + str(co[0]) + " "  + str(co[1]) + " " + str(co[2]) + " " + str(co[3])
    print (content)
    with open(log_file, 'a') as appender:
        appender.write(content + '\n')    

    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cosine_epo)
    scheduler_warmup = GradualWarmupSchedulerV2(optimizer, multiplier=10, total_epoch=warmup_epo, after_scheduler=scheduler_cosine)

    best_score = 0
    for epoch in range(n_epochs):

        print(time.ctime(), 'Epoch:', epoch)
        scheduler_warmup.step(epoch)        

        train_one_epoch(train_loader, model, optimizer, device=device)
        predictions, valid_targets = evaluate(valid_loader, model, device=device)
        predictions = np.array(predictions)
        valid_targets = np.array(valid_targets)
        
        un, co = np.unique(np.argmax(predictions, axis=1), return_counts=True)
        co = co / predictions.shape[0]
        xcontent = "Prediction Distribution 0, 1, 2, 3 -> " + str(co[0]) + " "  + str(co[1]) + " " + str(co[2]) + " " + str(co[3])

        scores, score = get_f1(valid_targets, predictions)
        roc_auc = roc_auc_score(valid_targets, predictions, multi_class='ovr')
        content = time.ctime() + ' ' + f'Fold {fold} Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, valid macro-f1(avg): {score:.4f}, valid auc: {(roc_auc):.4f}, valid macro-f1(none): {scores[0]:.4f}, valid macro-f1(infection): {scores[1]:.4f}, valid macro-f1(ischaemia): {scores[2]:.4f}, valid macro-f1(both): {scores[3]:.4f}.'
        content = content + " " + xcontent
        print(content)
        with open(log_file, 'a') as appender:
            appender.write(content + '\n')        

        if score > best_score:
            print('Valid Macro-F1(avg) increased ({:.6f} --> {:.6f}).  Saving model ...'.format(best_score, score))
            best_score = score
            torch.save(model.state_dict(),"/data0/Salman/alcer/models/" + baseline_name + '-' + str(fold) + '.pt')
    del model
    gc.collect()
    torch.cuda.empty_cache()

# %% [code]


# %% [code]


# %% [code]


# %% [code]


# %% [code]

