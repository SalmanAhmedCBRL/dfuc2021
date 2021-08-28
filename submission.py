# %% [code]
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
import matplotlib.pyplot as plt
import cv2
import time
from pylab import rcParams
rcParams['figure.figsize'] = 20,10
import warnings
warnings.simplefilter('ignore')
import gc
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
train = pd.read_csv("/data0/Salman/alcer/DFUC2021_train/train.csv")
file_names = np.sort(os.listdir("/data0/Salman/alcer/DFUC2021_test/"))
train.head()

# %% [code] {"scrolled":true}
valid_df = pd.DataFrame()
valid_df["image"] = file_names
valid_df["none"] = 0
valid_df["infection"] = 0
valid_df["ischaemia"] = 0
valid_df["both"] = 0
print (valid_df.shape)
valid_df.head()

# %% [code]
# valid_df["both"] = 1
# valid_df.to_csv("just3.csv", index=False)

# %% [code]
IMG = "/data0/Salman/alcer/DFUC2021_test/" + valid_df["image"]
IMG = IMG.values
T1 = valid_df["none"].values
T2 = valid_df["infection"].values
T3 = valid_df["ischaemia"].values
T4 = valid_df["both"].values

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
baseline_name = "tf_efficientnet_b0"

# %% [code]
class CustomModel(nn.Module):
    def __init__(self, model_name, pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=n)

    def forward(self, x):
        output = self.model(x)
        return output

# %% [code]
preds = []
for fold in range(0, N_FOLDS):
    model = CustomModel(baseline_name, pretrained=True)
    model.load_state_dict(torch.load("/data0/Salman/alcer/models/" + baseline_name + '-' + str(fold) + '.pt'))
    model.to(device)
    valid_dataset = ClassificationDataset(IMG, T1, T2, T3, T4, None)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=Batch_Size,shuffle=False, num_workers=n_workers)
    predictions, valid_targets = evaluate(valid_loader, model, device=device)
    predictions = np.array(predictions)
    preds.append(predictions)

# %% [code]
# fold = 1
# model = CustomModel(model_name=baseline_name, pretrained=True)
# model.load_state_dict(torch.load("/data0/Salman/alcer/models/" + baseline_name + '-' + str(fold) + '.pt'))
# model.to(device)
# valid_dataset = ClassificationDataset(IMG, T1, T2, T3, T4, None)
# valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=Batch_Size,shuffle=False, num_workers=n_workers)
# predictions1, valid_targets = evaluate(valid_loader, model, device=device)
# predictions1 = np.array(predictions1)

# %% [code]
predictions = (preds[0] + preds[1] + preds[2] + preds[3] + preds[4]) / 5.0
#predictions = preds[0]

# %% [code]
np.unique(np.round(predictions), return_counts=True)

# %% [code]
valid_df["none"] = predictions[:, 0]
valid_df["infection"] = predictions[:, 1]
valid_df["ischaemia"] = predictions[:, 2]
valid_df["both"] = predictions[:, 3]

# %% [code]
valid_df.head()

# %% [code]
valid_df.to_csv("efb089submission.csv", index=False)

# %% [code]
np.unique(np.round(predictions), return_counts=True)

# %% [code] {"scrolled":true}
np.unique(np.argmax(predictions, axis=1), return_counts=True)

# %% [code]
np.unique(np.argmax(predictions, axis=1), return_counts=True)

# %% [code]
np.unique(np.argmax(predictions, axis=1), return_counts=True)

# %% [code]
w = np.array([279, 143,  35,  43])
w = w / 500
w

# %% [code]
w = np.array([219, 229,  15,  37])
w = w / 500
w

# %% [code]
w = np.array([218, 223,  17,  42])
w = w / 500
w

# %% [code] {"scrolled":true}
df1 = pd.read_csv("v0submission.csv")
df2 = pd.read_csv("v3submission.csv")
df1.head()

# %% [code] {"scrolled":true}
# df1["none"] = 0.7*df1["none"] + 0.3*df2["none"]
# df1["infection"] = 0.7*df1["infection"] + 0.3*df2["infection"]
df1["ischaemia"] = 0.3*df1["ischaemia"] + 0.7*df2["ischaemia"]
# df1["both"] = 0.7*df1["both"] + 0.3*df2["both"]
df1.head()

# %% [code]
df1.to_csv("v3-0submission.csv", index=False)

# %% [code]


# %% [code]
df = pd.read_csv("efb089submission.csv")
df["both"] = df["both"] ** 0.55
df["ischaemia"] = df["ischaemia"] ** 2.2
df["infection"] = df["infection"] ** 0.4
df["none"] = df["none"] ** 1.6
df.head()

# %% [code]
np.unique(np.argmax(df[['none', 'infection', 'ischaemia', 'both']].values, axis=1), return_counts=True)[1] / df.shape[0]

# %% [code]
df.to_csv("moded_v0_89_test.csv", index=False)

# %% [code]
df = pd.read_csv("efb0-submission.csv")
df["both"] = df["both"] ** 0.55
df["ischaemia"] = df["ischaemia"] ** 2.2
df["infection"] = df["infection"] ** 0.4
df["none"] = df["none"] ** 1.6
np.unique(np.argmax(df[['none', 'infection', 'ischaemia', 'both']].values, axis=1), return_counts=True)

# %% [code]
df.to_csv("modedb0.csv", index=False)

# %% [code]


# %% [code]
df = pd.read_csv("efb0-submission.csv")
np.unique(np.argmax(df[['none', 'infection', 'ischaemia', 'both']].values, axis=1), return_counts=True)

# %% [code]
fig, axs = plt.subplots(2, 2)
v = df["none"].values
vv = v**1.6
axs[0, 0].plot(list(range(0, df.shape[0])), v[np.argsort(v)], 'b', label='Probability')
axs[0, 0].plot(list(range(0, df.shape[0])), vv[np.argsort(v)], 'g', label='Adjusted Probability p=1.6')
axs[0, 0].set_title('Control')

v = df["infection"].values
vv = v**0.4
axs[0, 1].plot(list(range(0, df.shape[0])), v[np.argsort(v)], 'b', label='Probability')
axs[0, 1].plot(list(range(0, df.shape[0])), vv[np.argsort(v)], 'g', label='Adjusted Probability p=0.4')
axs[0, 1].set_title('Infection')

v = df["ischaemia"].values
vv = v**2.2
axs[1, 0].plot(list(range(0, df.shape[0])), v[np.argsort(v)], 'b', label='Probability')
axs[1, 0].plot(list(range(0, df.shape[0])), vv[np.argsort(v)], 'g', label='Adjusted Probability p=2.2')
axs[1, 0].set_title('Ischaemia')

v = df["both"].values
vv = v**0.55
axs[1, 1].plot(list(range(0, df.shape[0])), v[np.argsort(v)], 'b', label='Probability')
axs[1, 1].plot(list(range(0, df.shape[0])), vv[np.argsort(v)], 'g', label='Adjusted Probability p=0.55')
axs[1, 1].set_title('Both')


for ax in axs.flat:
    ax.set(xlabel='example', ylabel='probablity')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()
    ax.legend()
fig.suptitle('Comparison of Softmax Probabilities and Bias Adjusted Probabilities for each class')

# %% [code]


# %% [code]


# %% [code]


# %% [code]


# %% [code]


# %% [code]
fig, axs = plt.subplots(2, 2)
v = df["none"].values
srt = np.argsort(v)
vv = v**1.6
axs[0, 0].plot(list(range(0, df.shape[0])), v[srt], 'b', label='Probability')
axs[0, 0].plot(list(range(0, df.shape[0])), vv[srt], 'g', label='Adjusted Probability p=1.6')
axs[0, 0].set_title('Control')

v = df["infection"].values
vv = v**0.4
axs[0, 1].plot(list(range(0, df.shape[0])), v[srt], 'b', label='Probability')
axs[0, 1].plot(list(range(0, df.shape[0])), vv[srt], 'g', label='Adjusted Probability p=0.4')
axs[0, 1].set_title('Infection')

v = df["ischaemia"].values
vv = v**2.2
axs[1, 0].plot(list(range(0, df.shape[0])), v[srt], 'b', label='Probability')
axs[1, 0].plot(list(range(0, df.shape[0])), vv[srt], 'g', label='Adjusted Probability p=2.2')
axs[1, 0].set_title('Ischaemia')

v = df["both"].values
vv = v**0.55
axs[1, 1].plot(list(range(0, df.shape[0])), v[srt], 'b', label='Probability')
axs[1, 1].plot(list(range(0, df.shape[0])), vv[srt], 'g', label='Adjusted Probability p=0.55')
axs[1, 1].set_title('Both')


for ax in axs.flat:
    ax.set(xlabel='example', ylabel='probablity')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()
    ax.legend()
fig.suptitle('Comparison of Softmax Probabilities and Bias Adjusted Probabilities for each class')

# %% [code]

