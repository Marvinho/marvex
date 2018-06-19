import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dataloaderwhalesiam import HumpbackWhaleDataset
import torchvision
from torchvision import models
from torchvision import transforms, utils
from torchvision.transforms import ToPILImage, Resize, RandomHorizontalFlip, RandomRotation, ToTensor, Compose, RandomGrayscale
from PIL import Image
import contrastiveLoss

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
      
        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(3, 6, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(6),
            nn.Dropout2d(p=.2),
            
            nn.ReflectionPad2d(1),
            nn.Conv2d(6, 12, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(12),
            nn.Dropout2d(p=.2),

            nn.ReflectionPad2d(1),
            nn.Conv2d(12, 24, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(24),
            nn.Dropout2d(p=.2),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(8*100*100, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 5)
        )

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2
    
    def loss(self, out1, out2, label):
        
        loss = ContrastiveLoss()
        #print(target)
        #target = target.squeeze()
        #target = target.type(torch.cuda.LongTensor)
        return loss(out1, out2, label)
        
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
    img_size = (600, 600)
    composed = Compose([ToPILImage(), Resize(img_size), RandomHorizontalFlip(), RandomGrayscale(p=0.5), 
    					RandomRotation(degrees = 30, center = None), ToTensor(), normalize])

    train_dataset = HumpbackWhaleDataset(csv_file = './train_no_nu_whales.csv', root_dir= "./train", transform = composed)
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


    net = Net()
    #num_ftrs = net.fc.in_features
    #net.classifier._modules["6"] = nn.Dropout(p = 0.75)
    #net.classifier._modules["7"] = nn.Linear(in_features = 4096, out_features = 4250)
    net.cuda()

    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    for epoch in range(0, 1000):
        lossSumm = 0
        print("Epoch {}".format(epoch))
        
        b = 0
        
        for data in train_loader:
            correct = 0
            b = b + 1
            image1, image2, label = data
            image1, image2, label = Variable(image1), Variable(image2), Variable(label)
            image1 = image1.cuda()
            image2 = image2.cuda()
            label = label.cuda()
            
            output1, output2 = net(image1, image2)
            #_,predict = torch.max(output.data, 1)
            #predict = predict.type(torch.FloatTensor)
            #predict = predict.cuda()
            #print(predict[0])
            #print(target[0].data[0])
            #for pos in range(0, len(target.data)):
            #    if(target[pos].data[0] == predict[pos]):
            #        correct = correct + 1
            #print(correct)
            #exit()
            #criterion = nn.CrossEntropyLoss()
            #target = target.squeeze()
            #target = target.type(torch.cuda.LongTensor)
            loss = loss(output1, output2, label)
            
            loss.backward()
            lossSumm = lossSumm + loss.data[0]
            optimizer.step()
            optimizer.zero_grad()
            
            print("batch:{}, loss:{:.4f}".format(b, loss.data[0]))
                  
            del loss, output1, output2, label
        if(epoch%50 == 0):
        	print("saved")
        	torch.save(net.state_dict(), "./model.pth")

        plotErrors(lossSumm/len(train_loader.dataset))
        torch.cuda.empty_cache()    