#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append('pytorchvideo')
from pytorchvideo.data import LabeledVideoDataset


# In[2]:


from glob import glob
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


# # Load video files into data frame. Create test/train data frames

# In[17]:


import json

f = open('C:\\Users\\bysan\\Downloads\\WLASL-master\\WLASL-master\\start_kit\\WLASL_v0.3.json')
data = json.load(f)
file_gloss_map = {}
print(type(data))
print(len(data))
print(data[1]['gloss'])

for video_id in data[1]['instances']:
    print(video_id['video_id'])

print(json.dumps(data[1], indent=2))


# In[ ]:


#Set up File Connections
#X=glob('filepath/*') but for every word, probably use data table
#label = [0]*len(X)+[1]*len(Y)+[2]*len(Z)
# parse the wlasl json and create data frame file vs gloss
df=pd.DataFrame(zip(X+Y+Z, label),columns=['file','gloss'])
#create a validation dataset that isn't trained, but is tested on
from sklearn.model_selection import train_test_split #downloading ski-learn in background
train_df,test_df=train_test_split(df,test_size=0.2,shuffle=True)
#70/20/10 for train, val, and test
len(train_df),len(test_df)


# # Define video augmentation functions

# In[ ]:


#augmentation process
from pytorchvideo.data import LabeledVideoDataset, make_clip_sampler, labeled_video_dataset

from pytorchvideo.transforms import (
    ApplyTransformToKey,         # Applies a transform to a specific key in the video dictionary
    Normalize,                  # Normalizes data
    RandomShortSideScale,       # Scales videos to have a random short side length
    UniformTemporalSubsample,   # Subsamples video frames uniformly over time
    Permute                     # Changes the order of dimensions in video tensors
)

from torchvision.transforms import (
    Compose,                    # Composes several transforms together
    Lambda,                     # Applies a custom lambda/function transform
    RandomCrop,                 # Randomly crops images or videos
    RandomHorizontalFlip,       # Randomly flips images or videos horizontally
    Resize                      # Resizes images or videos to a specific size
)

from torchvision.transforms._transforms_video import (
    CenterCropVideo,            # Crops the center portion of videos
    NormalizeVideo              # Normalizes video data
)


# In[ ]:


video_transform=Compose{[
    ApplyTransformToKey(key='video',
    transform=Compose([
        UniformTemporalSubsample(20),
        Lambda(lambda x:x/255),
        Normalize((0.45,0.45,0.45),(0.225,0.225,0.225))
        RandomShortSideScale(min_size=248,max_size=256),
        CenterCropVideo(224),

    ]),
    ),
])


# # Create train, test data set and corresponding data_loader

# In[ ]:


from torch.utils.data import DataLoader
train_dataset=labeled_video_dataset(train_df,
                                    clip_sampler=make_clip_sampler('random', 2),
                                     transform=video_transform,decode_audio=False)
train_loader=DataLoader(train_dataset,batch_size=5,num_workers=0,pin_memory=True)

#create test dataset/loader
test_dataset=labeled_video_dataset(test_df,
                                    clip_sampler=make_clip_sampler('random', 2),
                                     transform=video_transform,decode_audio=False)
test_loader=DataLoader(test_dataset,batch_size=5,num_workers=0,pin_memory=True)


# # Create a CNN model

# In[ ]:


#deal with later
batch=next(iter(loader))
batch.keys()
batch['video'].shape,batch['label'].shape


# In[ ]:


import torch.nn as nn
import torch
from pytorch_lightning import LightningModule, seed_everything, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint,LearningRateMonitor
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import classification_report
import torchmetrics


# In[ ]:


class OurModel(LightningModule):
    def __init__(self):
        super(OurModel,self).__init__()
        # model architecture
        self.video_model=torch.hub.load('facebookresearch/pytorchvideo', 'efficient_x3d_xs',pretrained=True)
        self.relu=nn.ReLU()
        self.linear=nn.Linear(400,1)

        self.lr=1e-3
        self.batch_idx_size=8 #Change based on Training Hardware Specs
        self.numworker=6
        #evaluation metric
        self.metric=torchmetrics.Accuracy()
        #loss function
        self.criterion=nn.BCEWithLogitsLoss()

    def forward(self,x):
        x=self.video_model(x)
        x=self.relu()
        x=self.linear(x)
        return x

    def configure_optimizers(self):
        opt=torch.optim.AdamW(params=self.parameters(),lr=self.lr)
        scheduler=CosineAnnealingLR(opt,T_0,10,eta_min=1e-6.last_epoch=-1)
        return { 'optimizer':opt,'lrscheduler':scheduler}

    def train_dataloader(self):
        dataset=labeled_video_dataset(train_df,
                                            clip_sampler=make_clip_sampler('random',2),
                                             transform=video_transform, decode_audio=False)
        loader=DataLoader(dataset,batch_size=self.batch_size,num_workers=self.numworker,pin_memory=True)
        return loader

    def training_step(self,batch,batch_idx):
        video,label=batch
        out=self(video)
        loss=self.criterion(out,label)
        metric=self.metric(out,label.to(torch.int64))
        return {'loss':loss, "metric":metric.detach()}

    def training_epoch_end(self, outputs):
        loss=torch.stack([x['loss'] for x in outputs]).mean().cpu().numpy().round(2)
        metric=torch.stack([x['metric'] for x in outputs]).mean().cpu().numpy().round(2)
        self.log('training_loss',loss)
        self.log('training_metric',metric)
        
    def test_dataloader(self):
        

