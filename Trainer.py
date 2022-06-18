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

class SEM_Depth_Dataset4(Dataset):
    """Sem Depth dataset."""

    def __init__(self, sem_path, depth_path, transform=None):
        """
        Args:
            sem_path (string): Path to the Sem images.
            depth_path (string): Path to the Sem images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        
        self.sem_path = sem_path
        self.depth_path = depth_path
        self.sem_list = np.sort(glob.glob(self.sem_path+ '/*.png'))
        
        if self.depth_path is not None:
            self.depth_list = np.sort(glob.glob(self.depth_path+ '/*.png'))
            
        self.transform = transform

    def __len__(self):
        return len(self.sem_list)

    def __getitem__(self, idx):

        if self.depth_path is None:
            img_path = self.sem_list[idx]
            img = Image.open(img_path)

            if self.transform is not None:
                img = self.transform(img)
            
            return {'base_img':img}
        
        else:
            depth_idx = idx//4
            img_path = self.sem_list[idx]
            label_path = self.depth_list[depth_idx]
            
            img = Image.open(img_path)
            label = Image.open(label_path)
            
            base_idx = idx%4
            
            sim_idx = idx//4*4
            sim_img1 = Image.open(self.sem_list[sim_idx])
            sim_img2 = Image.open(self.sem_list[sim_idx+1])
            sim_img3 = Image.open(self.sem_list[sim_idx+2])
            sim_img4 = Image.open(self.sem_list[sim_idx+3])
            
            
            if self.transform is not None:
                img = self.transform(img)
                label = self.transform(label)
                sim_img1 = self.transform(sim_img1)
                sim_img2 = self.transform(sim_img2)
                sim_img3 = self.transform(sim_img3)
                sim_img4 = self.transform(sim_img4)
            
            sim_img = torch.cat((sim_img1,sim_img2,sim_img3,sim_img4),0)

            return {'base_img':img, 'label':label, 'sim_img' : sim_img , 'base_idx': base_idx }
        
        
def make_modelname():
    now = datetime.datetime.now()
    year = str(now.year)[-2:]
    month = str(now.month) if now.month >= 10 else str(0) + str(now.month)
    day = str(now.day) if now.day >= 10 else str(0) + str(now.day)
    model_name = year + month + day
    return model_name


def evaluate(model, iterator, is_norm=False, is_rmse=False):
    loss_fn = torch.nn.MSELoss()
    model.eval()
    eval_loss = 0.0
    n_step = 0
    
    with torch.no_grad():
        for batch in iterator:
            img, label = batch['base_img'], batch['label']
            img = img.to(device)
            label = label.to(device)
            
            output = model(img.repeat(1,3,1,1))[:, :1, :, :]
            
            if is_norm:
                output = (output*0.5+0.5) * 255
                label = (label*0.5+0.5) * 255
            else:
                output = output * 255
                label = label * 255
            
            loss = torch.sqrt(loss_fn(output,label)) if is_rmse else loss_fn(output,label)
            eval_loss += loss.item()
            n_step += 1

    eval_loss = eval_loss / len(iterator.dataset)
    
    return eval_loss


def train(args, model, train_dataloader, valid_dataloader, model_name):

    early_stop_num = 0
    best_epoch = 0
    best_eval_loss = 99999999999999999999999999
    
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = torch.nn.MSELoss()
    #best_eval_loss = float('inf')
    
    tr_loss = 0
    eval_loss = 0
        
    tr_losses = []
    eval_losses = []
    
    print_batch = int(len(train_dataloader.dataset) / (args.train_batch_size * 2))
    tot_train_data_num, tot_eval_data_num = 0, 0
    num_step = 0
    
    for epoch in range(int(args.num_train_epochs)):
        model.zero_grad()
        model.train()
        optimizer.zero_grad()
        
        for step, tr_batch in enumerate(train_dataloader):
            # train
            tr_img, tr_label = tr_batch['base_img'], tr_batch['label']
            tr_img = tr_img.to(args.device)
            tr_label = tr_label.to(args.device)

            if model_name == 'autoEncoder':
                tr_output, train_encoder_vec = model(tr_img)
            elif model_name == 'unet':
                tr_output = model(tr_img.repeat(1,3,1,1))[:, :1, :, :]
              
            tmp_tr_loss = torch.sqrt(loss_fn(tr_output,tr_label))
            
            # contrastive learning
            if args.is_contrast and model_name == 'autoEncoder':
                # similar data
                sim_rand_choice = torch.randperm(4)[0]
                sim_img = tr_batch['sim_img'][:,sim_rand_choice,:,:].view(-1, 1, 66, 45).to(args.device)
                sim_img_encoder_vec = model(sim_img, only_encoding=True)
                sim_loss = torch.sqrt(loss_fn(sim_img_encoder_vec, train_encoder_vec))

                # contrastive data
                contrast_rand_perm = torch.randperm(len(tr_img))
                contrast_img = tr_img[contrast_rand_perm,...]
                contrast_img_encoder_vec = model(contrast_img, only_encoding=True)
                contrast_loss = -torch.sqrt(loss_fn(contrast_img_encoder_vec, train_encoder_vec))
                
                aux_loss = (sim_loss + contrast_loss)
            else:
                aux_loss = 0.0
            
            tot_tr_loss = tmp_tr_loss + args.aux_weight*aux_loss
            tot_tr_loss.backward()
            
            tr_loss += tmp_tr_loss.item() * 255
            
            # valid
            eval_batch = next(iter(valid_dataloader))
            eval_img, eval_label = eval_batch['base_img'], eval_batch['label']
            eval_img = eval_img.to(args.device)
            eval_label = eval_label.to(args.device)
            
            if model_name == 'autoEncoder':
                eval_output, _ = model(eval_img)
            elif model_name == 'unet':
                eval_output = model(eval_img.repeat(1,3,1,1))[:, :1, :, :]
            
            tmp_eval_loss = torch.sqrt(loss_fn(eval_output,eval_label))
            eval_loss += tmp_eval_loss.item() * 255
            
            optimizer.step()
            optimizer.zero_grad()
            
            num_step += 1
            
            # printing
            if step % print_batch == 0:
                tr_loss_print = tr_loss / num_step
                eval_loss_print = eval_loss / num_step

                current_time = datetime.datetime.now()
            
                if eval_loss_print < best_eval_loss:
                    best_eval_loss = eval_loss_print
                    torch.save(model.state_dict(), args.PATH)
                    best_epoch = epoch
                    early_stop_num = 0
                    print(f"[{current_time} "+ "Epoch: %3d, Batch: %3d ] Train_loss: %4.4f, Eval_loss: %4.4f > New best model"
                         % (epoch+1, step, tr_loss_print, eval_loss_print))
                else:
                    early_stop_num += 1
                    print(f"[{current_time} "+ "Epoch: %3d, Batch: %3d ] Train_loss: %4.4f, Eval_loss: %4.4f"
                         % (epoch+1, step, tr_loss_print, eval_loss_print))
                
                tr_loss, eval_loss = 0.0, 0.0
                tot_train_data_num, tot_eval_data_num = 0, 0
                num_step = 0
                
                tr_losses.append(tr_loss_print)
                eval_losses.append(eval_loss_print)
                
                info = {'tr_losses':tr_losses, 'eval_losses':eval_losses, 'best_eval_loss':best_eval_loss, 'best_epoch':best_epoch}
                
                if early_stop_num == args.early_stop_thres:
                    return info
    return info