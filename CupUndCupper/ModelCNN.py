# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 11:31:28 2018

@author: MRVN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, utils
from torchvision.datasets import ImageFolder
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import time


class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5)
        self.conv2 = nn.Conv2d(16, 32, kernel_size = 5)
        self.conv3 = nn.Conv2d(32, 32, kernel_size = 3)
        self.conv_dropout = nn.Dropout2d()
        self.dropout = nn.Dropout()
        self.fc1 = nn.Linear(32*5*5, 256)
        self.fc2 = nn.Linear(256, 43)
    
    
    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
#        print(x.shape)
        x = self.conv3(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
#        print(x.shape)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x)


    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features




device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

batch_size = 32
image_size = (64,64)
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

traindatapath = "F:\InformatiCup\GTSRB\Final_Training\Images"
testdatapath = "F:\InformatiCup\GTSRB\Final_Test\Images"

data_transforms = {
        "train": transforms.Compose([transforms.Resize(size = image_size), 
                            transforms.ToTensor(), 
                            transforms.Normalize(mean, std)]),
        "test": transforms.Compose([transforms.Resize(size = image_size), 
                            transforms.ToTensor(), 
                            transforms.Normalize(mean, std)])
    }

#print(data_transforms["train"])

full_data = ImageFolder(root = traindatapath, transform = data_transforms["train"])


train_size = int(0.8 * len(full_data))
test_size = len(full_data) - train_size
train_data, test_data = random_split(full_data, [train_size, test_size])

#print(len(train_data))

trainloader = DataLoader(dataset = train_data, batch_size = batch_size, shuffle = True)

testloader = DataLoader(dataset = test_data, batch_size = batch_size)

class_names = full_data.classes

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


# Get a batch of training data
inputs, classes = next(iter(trainloader))

# Make a grid from batch
out = utils.make_grid(inputs)

#imshow(out, title=[class_names[x] for x in classes])


model = Net()
model = model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

def training(model = model, criterion = criterion, optimizer = optimizer, num_epochs = 61):
    
    print("start training...")
    since = time.time()
    model.train()
    
    for epoch in range(num_epochs):
        timestart_epoch = time.time()
        print("Epoch {}/{}".format(epoch, num_epochs-1))
        running_loss = 0.0
        running_corrects = 0.0
        
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            #print("put inputs and labels on gpu...")
    
    
            # zero the parameter gradients
            optimizer.zero_grad()
    
            # forward + backward + optimize
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(predicted.data == labels.data)

        epoch_loss = running_loss / train_size
        epoch_accuracy = (running_corrects.item() / train_size) * 100
        print("Loss: {:.4f} Accuracy: {:.4f}%".format(epoch_loss, epoch_accuracy))
        
        time_elapsed = time.time() - timestart_epoch
        print("finished epoch in {:.0f}m {:.0f}s.".format(time_elapsed // 60, time_elapsed % 60))
        
        if(epoch%4 == 0 and epoch > 0):
            print()
            testing(epoch)
            print()
    
    time_elapsed = time.time() - since
    print("finished training in {:.0f}m {:.0f}s.".format(time_elapsed // 60, time_elapsed % 60))



def testing(epoch, model = model, criterion = criterion):
    print("start testing...")
    model.eval()
    with torch.no_grad():
        
        running_loss = 0.0
        running_corrects = 0.0
        
        for data in testloader:
            # get the inputs
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

    
            # forward + backward + optimize
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            loss = criterion(outputs, labels)
    
            # print statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(predicted.data == labels.data)
            

        epoch_loss = running_loss / test_size
        epoch_accuracy = (running_corrects.item() / test_size) * 100
        print("Loss: {:.4f} Accuracy: {:.4f}%".format(epoch_loss, epoch_accuracy))
    
#    print('Accuracy of the network on the test images: {}'.format(epoch_accuracy))
    print("finished testing.")
    print("saving the model...")
    torch.save(model.state_dict(), "saved_model_state_CNN_{}.pth".format(epoch))
    print("saved the model.")

#training()
#testing()
