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

from Trainer import *
from Model import *


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]= "1"

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

train_PATH = "./Train"
valid_PATH = "./Valid"
test_PATH = "./Test"

        
if __name__ == "__main__":
    # Args
    parser = argparse.ArgumentParser()
    parser.add_argument("-mn", "--model_name", default="autoEncoder") # autoEncoder, unet
    parser.add_argument("-in", "--is_norm", default=0)                # is_norm=1 : normalize with mean 0.5 and std 0.5
    parser.add_argument("-e", "--epochs", default=100)                # num of epochs
    parser.add_argument("-bs", "--batch_size", default=256)           # num of batches
    parser.add_argument("-lr", "--learning_rate", default=0.0001)     # learning rate
    parser.add_argument("-est", "--early_stop_thres", default=20)     # early stop condition. If there is no update on parameters n-times, terminates training.
    parser.add_argument("-d", "--device", default='cuda')             # cuda setting
    parser.add_argument("-id", "--is_dataAug", default=0)             # is_dataAug=1 : data augmentation ON (True)
    parser.add_argument("-ic", "--is_contrast", default=0)            # is_constrast=1 : contrastive learning ON (True)
    parser.add_argument("-aw", "--aux_weight", default=0)             # weight for constrastive learning auxiliary loss
    parser.add_argument("-ip", "--is_pretrained", default=0)          # is_pretrained=1 : use pre-trained model (only unet)
    parser.add_argument("-an", "--additional_name", default='') 
    
    args = Namespace()
    argparse_args = parser.parse_args()
    
    model_name = argparse_args.model_name
    args.is_norm = bool(int(argparse_args.is_norm))
    args.num_train_epochs = int(argparse_args.epochs)
    args.train_batch_size, args.valid_batch_size, args.test_batch_size = int(argparse_args.batch_size), int(argparse_args.batch_size), int(argparse_args.batch_size)
    args.learning_rate = float(argparse_args.learning_rate)
    args.early_stop_thres = int(argparse_args.early_stop_thres)
    args.device = argparse_args.device
    
    args.is_dataAug = bool(int(argparse_args.is_dataAug))
    args.is_contrast = bool(int(argparse_args.is_contrast))
    args.aux_weight = float(argparse_args.aux_weight)
    args.is_pretrained = bool(int(argparse_args.is_pretrained))
    args.additional_name = argparse_args.additional_name
    
    # Data Load
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5), (0.5))]) if args.is_norm else transforms.Compose([transforms.ToTensor()])

    #Train Dataset
    train_dataset = SEM_Depth_Dataset4(sem_path = "./Train/SEM", depth_path = "./Train/Depth", transform = transform)
    train_dataloader = DataLoader(dataset=train_dataset,
                            batch_size=args.train_batch_size,
                            shuffle=True)

    #Valid Dataset
    valid_dataset = SEM_Depth_Dataset4(sem_path = "./Validation/SEM", depth_path = "./Validation/Depth", transform = transform)
    valid_dataloader = DataLoader(dataset=valid_dataset,
                            batch_size=args.valid_batch_size,
                            shuffle=True)

    #Test Dataset
    test_dataset = SEM_Depth_Dataset4(sem_path = "./Test/SEM", depth_path = None, transform = transform)
    test_dataloader = DataLoader(dataset=test_dataset,
                            batch_size=args.test_batch_size,
                            shuffle=False)
    
    
    # Model Load

    training_config = f'norm-{args.is_norm}_lr-{args.learning_rate}_dataAug-{args.is_dataAug}_contrastLearn-{args.is_contrast}_auxWeight-{args.aux_weight}_is_pretrain-{args.is_pretrained}'
    current_date = make_modelname()
    args.PATH = f'./model/{model_name}_{training_config}_{current_date}_{args.additional_name}.pt'

    encoder_filter_nums = [3,3,3,3]
    decoder_filter_nums = [3,3,3,3,(4,3),3,3]
    encoder_channel_nums = [32,32,32,32]
    decoder_channel_nums = [32,32,32,32,32,32,32]

    if model_name == "autoEncoder":
        model_config = ModelConfig(encoder_filter_nums, decoder_filter_nums, encoder_channel_nums, decoder_channel_nums)
        model = DepthPrediction(model_config)
    else:
        model = torch.hub.load('milesial/Pytorch-UNet', 'unet_carvana', pretrained=args.is_pretrained, scale=0.5)
    model.to(args.device)
    
    
    # Train Model
    start = time.time()
    outputs = train(args, model,train_dataloader, valid_dataloader, model_name)
    end = time.time()
    print("Code execution time : " + str(end - start))
    
    
    with open(f'./Outputs/outputs_{model_name}_{training_config}_{current_date}_{args.additional_name}.pkl', 'wb') as f:
        pickle.dump(outputs, f)
        
        