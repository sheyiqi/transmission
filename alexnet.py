# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 10:26:44 2021

@author: yiqi.she
"""

## import AlexNet from pytorch official

from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision import datasets
import glob

import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
import torchvision.models as models
from torch.autograd import Variable
from torchvision.transforms import ToPILImage
import os
import torch.nn as nn
import torch.optim as optim
import torch.optim as optim
#from torchsummary import summary
from pytorch_model_summary import summary
import matplotlib.image as mpig


BATCH_SIZE = 128 # traning dataset per batch
EPOCHS = 90 # training times
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # training by CPU
SIZE = 227
PREDATADEAL = 0

##pre deal
if (PREDATADEAL == 1):
#    train_path = "D:/imagenet/ILSVRC2012/train_dataset"
#    count = 0
#    for i_train in os.listdir(train_path):
#        i_train_path = os.path.join(train_path, i_train)
#        for i_image in os.listdir(i_train_path):
#            i_image_path = os.path.join(i_train_path, i_image)
#            img = plt.imread(i_image_path)
#            if img.shape[0] < SIZE or img.shape[1] < SIZE or len(img.shape) < 3:
#                os.remove(str(i_image_path).replace('\\','/'))
#                count += 1
#    print ("there is "+ str(count) + " images` size < " + str(SIZE) + "!!!")
    
    test_path = "D:/imagenet/ILSVRC2012/validation_dataset"
    count = 0
    for i_test in os.listdir(test_path):
        i_test_path = os.path.join(test_path, i_test)
#        img = plt.imread(i_test_path)
#        if img.shape[0] < SIZE or img.shape[1] < SIZE or len(img.shape) < 3:
#            os.remove(str(i_test_path).replace('\\','/'))
#            count += 1
#    print ("there is "+ str(count) + " images` size < " + str(SIZE) + "!!!")
    
    exit()



## load ILSVRC2012
normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
transform = transforms.Compose([transforms.Pad(padding=0), 
                                transforms.RandomCrop(SIZE), # random center crop with size = 180*180
                                transforms.RandomHorizontalFlip(), # 50% probability flip pictures in random horizontal direction
                                transforms.ToTensor(), # trans data [0,255] to [0,1] (data/255), and tran H*W*C to C*H*W
                                normalize] # trans data from [0,1] to [-1,1]
                                 
                               )
train_dataset = datasets.ImageFolder("D:/imagenet/ILSVRC2012/train_100", transform=transform)
test_dataset = datasets.ImageFolder("D:/imagenet/ILSVRC2012/test_100", transform=transform)
#print (train_dataset)


# load train dataset & test dataset
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
#print (train_loader.dataset)
#print (test_loader.dataset)
#exit()
#print (test_loader.dataset.classes)
#print (test_loader.dataset.class_to_idx)
#print (test_loader.dataset.imgs)
##show image
#img = ToPILImage()(test_loader.dataset[0][0])
#img.show()

model = models.alexnet().to(DEVICE)
print (model)
param = list(model.named_parameters())
print (np.shape(list(model.parameters())[0]))
print (np.shape(list(model.parameters())[1]))
print (np.shape(list(model.parameters())[2]))
print (np.shape(list(model.parameters())[3]))
print (np.shape(list(model.parameters())[4]))
print (np.shape(list(model.parameters())[5]))
print (np.shape(list(model.parameters())[6]))
print (np.shape(list(model.parameters())[7]))
print (np.shape(list(model.parameters())[8]))
print (np.shape(list(model.parameters())[9]))
print (np.shape(list(model.parameters())[10]))
print (np.shape(list(model.parameters())[11]))
print (np.shape(list(model.parameters())[12]))
print (np.shape(list(model.parameters())[13]))
print (np.shape(list(model.parameters())[14]))
print (np.shape(list(model.parameters())[15]))

torch.nn.init.xavier_normal_(list(model.parameters())[0])
torch.nn.init.xavier_normal_(list(model.parameters())[2])
torch.nn.init.xavier_normal_(list(model.parameters())[4])
torch.nn.init.xavier_normal_(list(model.parameters())[6])
torch.nn.init.xavier_normal_(list(model.parameters())[8])
torch.nn.init.xavier_normal_(list(model.parameters())[10])
torch.nn.init.xavier_normal_(list(model.parameters())[12])
torch.nn.init.xavier_normal_(list(model.parameters())[14])

print ("init pass")
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
#optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.99))

#for batch_idx, (data, target) in enumerate(train_loader):
#    print (batch_idx)
#    print (len(data))
#    print (target)
#exit()


#### train
train_losses = list()
train_acces = list()
def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        #print (str(batch_idx) + " * " + str(BATCH_SIZE) + " images is training...")
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        _, pred = torch.max(output.data, 1) # calculation accuracy
        num_correct = (pred == target).sum().item()
        train_acc = num_correct / data.shape[0]
        
        train_loss = loss.item()
        train_losses.append(train_loss)
        train_acces.append(train_acc*100)
       
        if (batch_idx+1) % 10 == 0:
            print ('epoch:{}  [{}]  loss:{:.4f}  Accuracy: {}'.format(epoch+1, (batch_idx+1)*BATCH_SIZE, train_loss, train_acc))
        #if batch_idx % 100 == 0:
        #    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #        epoch, batch_idx * len(data), len(train_loader.dataset),
        #        100.0 * batch_idx / len(train_loader.dataset), loss.data[0]))
            
for epoch in range(EPOCHS):
    train(epoch)
if EPOCHS != 0:
    torch.save(model.state_dict(),"AlexNet.pth")


model1 = models.alexnet().to(DEVICE)
model1.load_state_dict(torch.load("AlexNet.pth"))
#### test
test_losses = list()
test_acces = list()
model1.eval()
with torch.no_grad():
    for batch_idx, (data, target) in enumerate(test_loader):

        
        
        data, target = Variable(data), Variable(target)
        output= model1(data)

        test_loss = criterion(output, target)
        test_losses.append(test_loss)
        
        _, test_pred = torch.max(output.data, 1)
        #print (str(batch_idx) + " *1 images is validating...")
        #print ("this picture is " + str(target))
        #print ("the predicted is " + str(test_pred))
        #img = ToPILImage()(data[0])
        #img.show()
        num_correct = (test_pred == target).sum().item()
        #print (batch_idx)
        test_acc = num_correct / data.shape[0]
        #print (data.shape[0])
        
        test_losses.append(test_loss)
        test_acces.append(test_acc*100)
        
        print ('avg loss: {:.4f}  avg acc: {}'.format(np.mean(test_losses), np.mean(test_acces)))
       
