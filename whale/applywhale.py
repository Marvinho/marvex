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
from trainWhales import Net
import pandas as pd

csv_file = ('./train.csv')
labels_frame = pd.read_csv(csv_file)
img_size = (224, 224)
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

net = models.alexnet(pretrained=True)
net.load_state_dict(torch.load("./model.pth"))
test_transf = Compose([ToPILImage(), Resize(img_size), ToTensor(), normalize])
test_dataset = HumpbackWhaleDataset(csv_file = './sample_submission.csv', root_dir= "./test", transform = test_transf)
#test_dataset = TitanicDataset(csvFile = 'test.csv')
arr = set()
for i in range(len(labels_frame)):
    arr.add(labels_frame.iloc[i,1])
arr = list(arr)

test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                           batch_size = 1,
                                           shuffle = False,
                                           num_workers = 4
                                           )
b = 1
result = [[]]
for data in test_loader:
	
	img_name = pd.read_csv('./sample_submission.csv').iloc[b-1, 0]
	image, label = data
	out = net(image)
	_, top5 = torch.topk(out, k = 5)
	result.append([img_name, "{} {} {} {} {}".format(str(arr[top5[0,0].item()]), 
													 str(arr[top5[0,1].item()]), 
													 str(arr[top5[0,2].item()]), 
													 str(arr[top5[0,3].item()]),
													 str(arr[top5[0,4].item()]))])
	b = b+1
	if(b % 100 == 0):
		print(b)

import csv
myFile = open("submission.csv","w")
with myFile:
	writer = csv.writer(myFile)
	writer.writerows(result)

	
