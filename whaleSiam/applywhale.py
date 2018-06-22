import torch
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2
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
from trainWhalesSiam import Net
import pandas as pd
import os
import numpy as np
import gc
root_dir = "./test"
csv_file = ('./oneshot.csv')
labels_frame1 = pd.read_csv(csv_file)
labels_frame2 = pd.read_csv("./sample_submission.csv")
img_size = (100, 100)
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

net = Net()
#net = net.cuda()
net.load_state_dict(torch.load("./model.pth"))
test_transf = Compose([ToPILImage(), Resize(img_size), ToTensor(), normalize])
result = []
for i in range(len(labels_frame2)):
	dists = []
	img_name1 = os.path.join("./test", labels_frame2.iloc[i, 0])
	image1 = Image.open(img_name1)
	image1 = image1.convert("RGB")
	image1 = np.array(image1)
	image1 = test_transf(image1)
	image1 = image1.unsqueeze(0)
	#image1 = image1.cuda()
	for j in range(len(labels_frame1)):
		img_name2 = os.path.join("./train", labels_frame1.iloc[j, 0])
		image2 = Image.open(img_name2)
		image2 = image2.convert("RGB")
		image2 = np.array(image2)
		image2 = test_transf(image2)
		image2 = image2.unsqueeze(0)
		#image2 = image2.cuda()
		out1, out2 = net(image1, image2)
		euclidean_distance = F.pairwise_distance(out1, out2)
		#print(euclidean_distance.data[0][0])
		dists.append([j, euclidean_distance.data[0][0]])
		del out1, out2, image2, img_name2, euclidean_distance
		torch.cuda.empty_cache()
		gc.collect()
		#if(j == 5):

		#	break
		if(j%100 == 0):
			print(j)

	dists.sort(key=lambda dist: dist[1])
	print("i: {}, img_name1: {}, top1: {}".format(i, labels_frame2.iloc[i,0], str(labels_frame1.iloc[dists[0][0],1])))
	result.append([labels_frame2.iloc[i,0], "{} {} {} {} {}".format(str(labels_frame1.iloc[dists[0][0],1]), 
    												str(labels_frame1.iloc[dists[1][0],1]), 
    												str(labels_frame1.iloc[dists[2][0],1]), 
    												str(labels_frame1.iloc[dists[3][0],1]), 
    												str(labels_frame1.iloc[dists[4][0],1]))])
	#print(result)
	#exit()
	del image1
	gc.collect()
	torch.cuda.empty_cache()



import csv
myFile = open("submission.csv","w")
with myFile:
	writer = csv.writer(myFile)
	writer.writerows(result)