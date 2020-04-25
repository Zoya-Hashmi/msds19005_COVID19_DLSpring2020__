
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from tqdm import tqdm
import torch.nn.functional as F

from utils import train,evaluate,initialize_model,load_data

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir',default='/content/drive/My Drive/Assignment 5 Dataset',help='Add the link to the directory that holds your data')
parser.add_argument('--batch_size',type =int,default=64,help='batch size for data to be loaded')
parser.add_argument('--model',default='vgg16',help='backbone model',choices=['vgg16','res18'])
parser.add_argument('--freeze',default='none',help='freeze none, partial or all conv layers',choices=['none','partial','all'])
parser.add_argument('--epochs',type =int,default =50,help='number of epochs to train')
parser.add_argument('--lr',type =float,default =1e-5,help='learning rate')
parser.add_argument('--pretrained_weights',help='path to pretrained weights')
parser.add_argument('--mode',default = '',help='train or evaluate',choices=['train','evaluate'])
parser.add_argument('--save_dir',default = '',help='directory to save checkpoints')

args = parser.parse_args(['--data_dir','/content/drive/My Drive/Assignment 5 Dataset','--batch_size','64','--model','res18','--freeze','all','--epochs','50','--lr','1e-5','--pretrained_weights','/content/drive/My Drive/TrainedModels/vgg16_ft_87.13_0.26_15.pth','--mode','train','--save_dir','/content'])

data_dir = args.data_dir
batch_size = args.batch_size
model= args.model
freeze=args.freeze
Epochs = args.epochs
lr = args.lr
pretrained_weights=args.pretrained_weights
mode= args.mode
save_dir = args.save_dir

trainloader,valloader,testloader = load_data(data_dir,batch_size)
net = initialize_model(model,freeze)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)
    
if mode == 'train':  
    train_loss,val_loss,train_acc,val_acc,learning_rates = train(net,trainloader,valloader,lr,save_dir)

elif mode=='evaluate':
    net.load_state_dict(torch.load(pretrained_weights)['state_dict'])
    evaluate(net,testloader)