import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dataloaderwhale import HumpbackWhaleDataset
import torchvision
from torchvision import models
from torchvision import transforms, utils
from torchvision.transforms import ToPILImage, Resize, RandomHorizontalFlip, RandomRotation, ToTensor, Compose, RandomGrayscale
from PIL import Image

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
      
        # an affine operation: y = Wx + b
        self.conv1 = nn.Conv2d(in_channels = 3,out_channels = 6, kernel_size = 5, stride = 2)
        self.conv2 = nn.Conv2d(in_channels = 6,out_channels = 12, kernel_size = 5, stride = 2)
        self.conv3 = nn.Conv2d(in_channels = 12,out_channels = 24, kernel_size = 5, stride = 2)
        self.conv4 = nn.Conv2d(in_channels = 24,out_channels = 36, kernel_size = 5, stride = 2)
        self.fc1 = nn.Linear(in_features = 59*34*36, out_features = 4251)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))
        #print(x.shape)
        x = x.view(-1, 59*34*36)
        
        x = self.fc1(x)
        
        return x
    
    def loss(self,x, target):
        
        loss = nn.CrossEntropyLoss()
        #print(target)
        target = target.squeeze()
        target = target.type(torch.cuda.LongTensor)
        return loss(x, target)
        
if __name__ == "__main__":
    trainErrsTotal = []
    testErrsTotal = []

    def plotErrors( trainErrs ):

        trainErrsTotal.append( trainErrs )
            #testErrsTotal.append( testErrs )

        plt.clf()
        
        plt.plot( trainErrsTotal, '-', label = "train total", color = (0.5,0,0.8) )
        

            #plt.plot( testErrsTotal, '-', label = "test total", color = (0.5,0.8,0) )

        plt.yscale('log')
        plt.grid(True)
        plt.legend()
        plt.savefig( "./errors" )
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    img_size = (224, 224)
    composed = Compose([ToPILImage(), Resize(img_size), RandomHorizontalFlip(), RandomGrayscale(p=0.5), 
    					RandomRotation(degrees = 30, center = None), ToTensor(), normalize])

    train_dataset = HumpbackWhaleDataset(csv_file = './train.csv', root_dir= "./train", transform = composed)
    #test_dataset = TitanicDataset(csvFile = 'test.csv')

    train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                               batch_size = 100,
                                               shuffle = True,
                                               num_workers = 4
                                               )
    #test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
    #                                           batch_size = 50,
    #                                           shuffle = True,
    #                                           num_workers = 4
    #                                           )


    net = models.alexnet(pretrained=True)
    #num_ftrs = net.fc.in_features
    net.classifier._modules["6"] = nn.Dropout(p = 0.75)
    net.classifier._modules["7"] = nn.Linear(in_features = 4096, out_features = 4250)
    net.cuda()

    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum = 0.1)

    for epoch in range(0, 1000):
        lossSumm = 0
        print("Epoch {}".format(epoch))
        
        b = 0
        
        for data in train_loader:
            correct = 0
            b = b + 1
            image, target = data
            image, target = Variable(image), Variable(target)
            image = image.cuda()
            target = target.cuda()
            
            output = net(image)
            _,predict = torch.max(output.data, 1)
            predict = predict.type(torch.FloatTensor)
            predict = predict.cuda()
            #print(predict[0])
            #print(target[0].data[0])
            for pos in range(0, len(target.data)):
                if(target[pos].data[0] == predict[pos]):
                    correct = correct + 1
            #print(correct)
            #exit()
            criterion = nn.CrossEntropyLoss()
            target = target.squeeze()
            target = target.type(torch.cuda.LongTensor)
            loss = criterion(output, target)
            
            loss.backward()
            lossSumm = lossSumm + loss.data[0]
            optimizer.step()
            optimizer.zero_grad()
            
            print("batch:{}, loss:{:.4f}, accuracy:{:.1f}".format(b, loss.data[0], correct / target.size(0) * 100))
                  
            del loss, output, target
        if(epoch%50 == 0):
        	print("saved")
        	torch.save(net.state_dict(), "./model.pth")

        plotErrors(lossSumm/len(train_loader.dataset))
        torch.cuda.empty_cache()    
