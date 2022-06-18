import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import glob
from torchvision import transforms
from argparse import Namespace
from tqdm.notebook import tqdm
import datetime
import time
import argparse
import pickle

class ModelConfig:
    def __init__(self, encoder_filter_nums, decoder_filter_nums, encoder_channel_nums, decoder_channel_nums):
        assert len(encoder_filter_nums) == 4
        assert len(encoder_channel_nums) == 4
        
#         assert len(decoder_filter_nums) == 6
#         assert len(decoder_channel_nums) == 6
        
        self.encoder_filter_nums = encoder_filter_nums
        self.decoder_filter_nums = decoder_filter_nums
        
        self.encoder_channel_nums = encoder_channel_nums
        self.decoder_channel_nums = decoder_channel_nums

        
class Encoder(nn.Module):
    def __init__(self, model_config):
        super(Encoder,self).__init__()
        
        self.model_config = model_config
        self.filter_nums = model_config.encoder_filter_nums
        self.channel_nums = model_config.encoder_channel_nums
        
        self.layer1 = nn.Sequential(
                        nn.Conv2d(1, self.channel_nums[0], self.filter_nums[0], padding='same'),                            
                        nn.ReLU(),
                        nn.BatchNorm2d(self.channel_nums[0]),
                        nn.Conv2d(self.channel_nums[0], self.channel_nums[1], self.filter_nums[1], padding='same'),          
                        nn.ReLU(),
                        nn.BatchNorm2d(self.channel_nums[1]),
                        nn.MaxPool2d(2,2)     
        )
        self.layer2 = nn.Sequential(
                        nn.Conv2d(self.channel_nums[1], self.channel_nums[2], self.filter_nums[2], padding='same'),                            
                        nn.ReLU(),
                        nn.BatchNorm2d(self.channel_nums[2]),
                        nn.Conv2d(self.channel_nums[2], self.channel_nums[3] ,self.filter_nums[3], padding='same'),          
                        nn.ReLU(),
                        nn.BatchNorm2d(self.channel_nums[3]),
                        nn.MaxPool2d(2,2)
        )

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        return out
    
    

class Decoder(nn.Module):
    def __init__(self, model_config):
        super(Decoder,self).__init__()
        
        self.model_config = model_config
        self.filter_nums = model_config.decoder_filter_nums
        self.channel_nums = model_config.decoder_channel_nums
        
        self.deconv_layer1 = nn.Sequential(
                        nn.ConvTranspose2d(self.channel_nums[0], self.channel_nums[1], self.filter_nums[0], stride=2), 
                        nn.ReLU(),
                        nn.BatchNorm2d(self.channel_nums[1]),
                        nn.ConvTranspose2d(self.channel_nums[1], self.channel_nums[2], self.filter_nums[1], stride=2), 
                        nn.ReLU(),
                        nn.BatchNorm2d(self.channel_nums[2])
        )
        self.deconv_layer2 = nn.Sequential(
                        nn.ConvTranspose2d(self.channel_nums[2], self.channel_nums[3], self.filter_nums[2], stride=1),  
                        nn.ReLU(),
                        nn.BatchNorm2d(self.channel_nums[3]),
                        nn.ConvTranspose2d(self.channel_nums[3], self.channel_nums[4], self.filter_nums[3], stride=1),
                        nn.ReLU(),
                        nn.BatchNorm2d(self.channel_nums[4]),
        )
        self.conv_layer = nn.Sequential(
                        nn.Conv2d(self.channel_nums[4], self.channel_nums[5], self.filter_nums[4], padding=(1,0)),                            
                        nn.ReLU(),
                        nn.BatchNorm2d(self.channel_nums[5]),
                        nn.Conv2d(self.channel_nums[5], self.channel_nums[6], self.filter_nums[5]),          
                        nn.ReLU(),
                        nn.BatchNorm2d(self.channel_nums[6]),
                        nn.Conv2d(self.channel_nums[6], 1, self.filter_nums[6]),          
                        nn.ReLU(),
        )
        
    def forward(self,x):
        out = self.deconv_layer1(x)
        out = self.deconv_layer2(out)
        out = self.conv_layer(out)
        return out
    
    
    
class DepthPrediction(nn.Module):
    def __init__(self, model_config):
        super(DepthPrediction, self).__init__()
        
        self.encoder = Encoder(model_config)
        self.decoder = Decoder(model_config)

    def forward(self, sem, only_encoding=False):
        if only_encoding:
            hidden_rep = self.encoder(sem)
            return hidden_rep
        
        else:    
            hidden_rep = self.encoder(sem)
            depth_pred = self.decoder(hidden_rep)
            return depth_pred, hidden_rep
    
    
    
    