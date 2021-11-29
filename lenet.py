# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 09:53:39 2021

@author: yiqi.she
"""
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.optim as optim
#from torchsummary import summary
from pytorch_model_summary import summary

torch.set_printoptions(precision=8, threshold=1000, edgeitems=1000, linewidth=10000)
numpy.set_printoptions(threshold=numpy.inf, linewidth=numpy.inf)

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,),(0.3081,))])
#transform = transforms.Compose([transforms.ToTensor()])
BATCH_SIZE = 100 # traning dataset per batch
EPOCHS = 1 # training times
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # training by CPU

train_dataset = torchvision.datasets.MNIST(root='./MNIST', train=True, download=True, transform=transform)

test_dataset = torchvision.datasets.MNIST(root='./MNIST', train=False, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

testloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

##############################################################################
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
    def forward(self, x):
        #in_size = x.size(0)
        x = self.conv1(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    

##############################################################################
model = ConvNet().to(DEVICE)
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()


def normalized_data(images):
    new_images = numpy.full(shape=(BATCH_SIZE,1,32,32),fill_value=images[0][0][0][0])
    new_images = torch.from_numpy(new_images)
    for i in range(len(images)):
        new_images[i][0][2:30,2:30] = images[i]
    #print (new_images)
    return new_images
    


####train
train_losses = list()
train_acces = list()
for epoch in range(EPOCHS):
    #break
    model.train()
    for i, (images, labels) in enumerate(trainloader):
        #print (images.size())
        images = normalized_data(images)
        #print (images.size()) # size length = batch
        #print (labels.size())
        optimizer.zero_grad()
        out = model(images)
        
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        
        _, pred = torch.max(out.data, 1) # calculation accuracy
        num_correct = (pred == labels).sum().item()
        train_acc = num_correct / images.shape[0]
        
        train_loss = loss.item()
        train_losses.append(train_loss)
        train_acces.append(train_acc*100)
        
        if (i+1) % 10 == 0:
            print ('epoch:{}  [{}]  loss:{:.4f}  Accuracy: {}'.format(epoch+1, (i+1)*BATCH_SIZE, train_loss, train_acc))
            
####test
test_losses = list()
test_acces = list()
model.eval()
with torch.no_grad():
    for i, (images, labels) in enumerate(testloader):
        images = normalized_data(images)
        
        out = model(images)
        test_loss = criterion(out, labels)
        test_losses.append(test_loss)
        
        _, test_pred = torch.max(out.data, 1)
        num_correct = (test_pred == labels).sum().item()
        test_acc = num_correct / images.shape[0]
        
        test_losses.append(test_loss)
        test_acces.append(test_acc*100)
        
        
        print ('avg loss: {:.4f}  avg acc: {}'.format(numpy.mean(test_losses), numpy.mean(test_acces)))
       
#print (images.size())
#print (torchvision.utils.make_grid(images))
#print (numpy.transpose(torchvision.utils.make_grid(images,nrow=10), (1,2,0)).size())
plt.axis('off')
plt.imshow(numpy.transpose(torchvision.utils.make_grid(images, nrow=10, padding=2), (1,2,0)))
print (labels.size())
print (labels.resize(10,10))

print (test_pred.data.resize(10,10))

print (labels.resize(10,10)-test_pred.data.resize(10,10))


#def write_file():


#print (model.state_dict())
with open ('LeNet-5/conv1.weight','w') as fw:
    fw.write(str(model.conv1.weight.detach().numpy()))

with open ('LeNet-5/conv1.bias','w') as fw:
    fw.write(str(model.conv1.bias.detach().numpy()))
    
with open ('LeNet-5/conv2.weight','w') as fw:
    fw.write(str(model.conv2.weight.detach().numpy()))

with open ('LeNet-5/conv2.bias','w') as fw:
    fw.write(str(model.conv2.bias.detach().numpy()))

with open ('LeNet-5/fc1.weight','w') as fw:
    fw.write(str(model.fc1.weight.detach().numpy()))

with open ('LeNet-5/fc1.bias','w') as fw:
    fw.write(str(model.fc1.bias.detach().numpy()))

with open ('LeNet-5/fc2.weight','w') as fw:
    fw.write(str(model.fc2.weight.detach().numpy()))

with open ('LeNet-5/fc2.bias','w') as fw:
    fw.write(str(model.fc2.bias.detach().numpy()))

with open ('LeNet-5/fc3.weight','w') as fw:
    fw.write(str(model.fc3.weight.detach().numpy()))

with open ('LeNet-5/fc3.bias','w') as fw:
    fw.write(str(model.fc3.bias.detach().numpy()))

with open ('LeNet-5/LeNet-5.model','w') as fw:
    fw.write(str(model))
    fw.write(str(summary(model, torch.zeros(1, 1, 32, 32))))
    
with open ('LeNet-5/testset.data','w') as fw:
    #print (images[0][0].size())
    fw.write(str(images[0][0].numpy()))
    #print (len(images[0][0].numpy()))
    #print (images[0][0].numpy())
    print (images[0][0].size())
    out = model(images[0][0].resize(1,1,32,32))
    print ("---")
    print (out)
    plt.imshow(numpy.transpose(torchvision.utils.make_grid(images[0][0], nrow=1, padding=2), (1,2,0)))
    
  #  for i in range(len(images[0][0].numpy())):
        #print (images[0][0].numpy()[i])
    #    fw.write(str(images[0][0].numpy()[i]))
#print (model.conv1.weight)
#print (model.conv1.bias)

#print (model.conv2.weight)
#print (model.conv2.bias)

#print (model.fc1.weight)
#print (model.fc1.bias)

#print (model.fc2.weight)
#print (model.fc2.bias)

#print (model.fc3.weight)
#print (model.fc3.bias)
#print (model.summary())

#print (model)
#print (summary(model, torch.zeros(100, 1, 32, 32)))

