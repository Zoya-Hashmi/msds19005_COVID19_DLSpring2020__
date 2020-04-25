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

def load_data(data_dir,batch_size):
    import torch
    import torchvision
    from torchvision import datasets, transforms

    #Define transforms for the training data and testing data
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                          transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.4905, 0.4905, 0.4906],
                                                                [0.2494, 0.2494, 0.2494])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.4905, 0.4905, 0.4906],
                                                              [0.2494, 0.2494, 0.2494])])


    #pass transform here-in
    train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
    val_data = datasets.ImageFolder(data_dir + '/validation', transform=test_transforms)
    test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)

    #data loaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valloader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)
    return trainloader,valloader,testloader

def initialize_model(model,freeze):
    if model == 'res18':
        res18 = models.resnet18(pretrained=True)
        fc1_in = res18.fc.in_features

        if freeze == 'all':
            for param in res18.parameters():
                param.requires_grad=False
        elif freeze == 'partial':
            for param in list(res18.parameters())[:-17]:
                param.requires_grad=False
        elif freeze == 'none':
            pass


    elif model == 'vgg16':
        vgg16 == models.vgg16(pretrained=True)
        fc1_in = vgg16.classifier[0].in_features

        if freeze == 'all':
            for param in vgg16.features.parameters():
                param.requires_grad = False
        elif freeze == 'partial':
            for param in list(vgg16.features.parameters())[:-6]:
                param.requires_grad = False
        elif freeze == 'none':
            pass


    fc1_out = 5*10+100
    fc2_in = fc1_out
    fc2_out = 2
    features = [nn.Linear(fc1_in,fc1_out,bias=True),nn.ReLU(inplace=True),nn.Dropout(p=0.5,inplace=False),nn.Linear(fc2_in,fc2_out,bias=True)]

    if model=='res18':
        res18.fc = nn.Sequential(*features)
        return res18
    if model == 'vgg16':
        vgg16.classifier = nn.Sequential(*features)
        if pretarined_weights != None:
          vgg16.load_state_dict(torch.load(pretrained_weights)['state_dict'])
        return vgg16


def train(model,trainoader,valloader,lr,save_dir):

  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)

  epoch_tacc = []
  epoch_tloss = []
  epoch_vacc= []
  epoch_vloss = []
  val_loss_min = 1e6
  val_acc_prev = 0.0
  epoch_lr = [] 

  # val_loss_min= 1e6

  for epoch in range(Epochs):  # loop over the dataset multiple times
      epoch_lr.append(lr)
      val_accuracy =  0.0
      train_accuracy = 0.0

      running_loss = 0.0
      # pbar = tqdm(total = len(trainloader))
      
      # pbar = tqdm(len(trainloader))
      
      
      model.train()
      pbar = tqdm(enumerate(trainloader),position=0,leave=True)
      # for i, data in pbar:
      for i,data in pbar:
          # get the inputs
          inputs, labels = data
          inputs, labels = inputs.to(device), labels.to(device)

          # zero the parameter gradients
          optimizer.zero_grad()
          # In PyTorch, we need to set the gradients to zero before starting to do backpropragation 
          # because PyTorch accumulates the gradients on subsequent backward passes. 
          # This is convenient while training RNNs. 
          # So, the default action is to accumulate the gradients on every loss.backward() call

          # forward + backward + optimize
          outputs = model(inputs)
          loss = criterion(outputs, labels)   #----> compute loss
          loss.backward()                     #----> backward pass
          optimizer.step()                    #----> weights update

          running_loss += loss.item()
          pred = torch.argmax(F.softmax(outputs),dim=1)
          correct = (pred==labels).float().sum()
          accuracy = correct*100/inputs.shape[0]
          train_accuracy += correct


          pbar.set_description(
              'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {:.1f}%'.format(
                  epoch, i * len(inputs), len(trainloader.dataset),
                  100. * i / len(trainloader),
                  loss.data.item(),accuracy),refresh=False)
          
      print("\nTraining Loss of Epoch ",epoch," is :",running_loss)
      tacc= train_accuracy*100/len(trainloader.dataset)
      print("Training Accuracy of Epoch ",epoch," is :",tacc.item(),"\n\n")
      

      epoch_tacc.append(tacc.item())
      epoch_tloss.append(running_loss)

      val_loss = 0.0
      model.eval()
      pbar = tqdm(enumerate(valloader),position=0,leave=True)
      for i,data in pbar:
          # get the inputs
          inputs, labels = data
          inputs, labels = inputs.to(device), labels.to(device)

          outputs = model(inputs)               #----> forward pass
          loss = criterion(outputs, labels)   #----> compute loss
  
          # print statistics
          val_loss += loss.item()
          pred = torch.argmax(F.softmax(outputs),dim=1)
          correct = (pred==labels).float().sum()
          accuracy = correct*100/inputs.shape[0]
          val_accuracy += correct
          
          pbar.set_description(
              'Validation Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {:.1f}%'.format(
                  epoch, i * len(inputs), len(valloader.dataset),
                  100. * i / len(valloader),
                  loss.data.item(),accuracy),refresh=False)
          
      print("\nValidation Loss of Epoch ",epoch," is :",val_loss)
      acc = val_accuracy*100/len(valloader.dataset)
      print("Validation Accuracy of Epoch ",epoch," is :",acc.item(),"\n\n")

      epoch_vloss.append(val_loss)
      epoch_vacc.append(acc.item())

      if val_loss < val_loss_min:
          checkpoint = {
                  'epoch': epoch + 1,
                  'valid_loss_min': val_loss,
                  'state_dict': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
              }

          val_loss_min = val_loss

          torch.save(checkpoint, osp.join(save_dir,'ckpt_{:.2f}_{:.2f}_{}.pth'.format(acc,loss,epoch)))



      delta = abs(acc.item() - val_acc_prev)
      val_acc_prev = acc.item()

      if (delta < 0.4)and(lr<0.1):
          lr = lr*10
          optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

      
  print('Finished Training')
  return epoch_tloss,epoch_vloss,epoch_tacc,epoch_vacc,epoch_lr  

# from sklearn.metrics import confusion_matrix
# import pandas as pd
# import itertools


def evaluate(model,loader):
  ## CONFUSION MATRIX ##
  
  correct = 0
  total = 0
  predicted = []
  gt = []
  with torch.no_grad():
      for data in loader:
          images, labels = data
          images, labels = images.to(device), labels.to(device)
          outputs = model(images)
          pred = torch.argmax(outputs,dim=1)
          correct += (pred==labels).float().sum()
          
          predicted.extend(torch.Tensor.cpu(pred).numpy())
          gt.extend(torch.Tensor.cpu(labels).numpy())
          total += labels.size(0)
          
  print('Accuracy: %d %%' % (
      100 * correct / total))

  # cfm = confusion_matrix(gt, predicted)
  # recall = cfm[1,1]/cfm[1,:].sum()
  # precision = cfm[1,1]/cfm[:,1].sum()
  # f1 = (2*precision*recall)/(precision+recall)
  # print("F1 SCORE: ",f1)
  