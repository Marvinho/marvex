# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 11:20:37 2018

@author: MRVN
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, utils, models
from torch.autograd.gradcheck import zero_gradients
from PIL import Image
from torch.autograd import Variable
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
import ModelCNN
import time






model = ModelCNN.Net()
#model = models.resnet18(pretrained=True)

epsilons = [0, .05, .1, .15, .2, .25, .3]
pretrained_model = "saved_model_state_CNN_final.pth"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

#image_name = "12629.ppm"
image_size = (64,64)
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
preprocess = transforms.Compose([transforms.Resize(size = image_size), 
                            transforms.ToTensor(), 
                            transforms.Normalize(mean, std)])

    
# =============================================================================
# image_tensor = Image.open(image_name) 
#     
# image_tensor = preprocess(image_tensor)
# image_tensor.requires_grad_()  
# image_tensor = image_tensor.unsqueeze(0)
# =============================================================================


testdatapath = "F:\InformatiCup\GTSRB_Python_code\Python code for GTSRB\Images"
test_data = ImageFolder(root = testdatapath, transform = preprocess)
testloader = DataLoader(dataset = test_data)





def loadModel(model = model):
    
    print("loading the model...")    
    model.load_state_dict(torch.load(pretrained_model))
    model.eval()
    print(model)
    print("loaded the model.")


def createImage(random = True, color = 0):
    
    w, h = 64, 64      # w * h
    print("creating image...")
    if(random == True):        
        test_image = np.random.randint(256, size=(w, h, 3), dtype=np.uint8)
        test_image = Image.fromarray(test_image, 'RGB')
    else:
        test_image = Image.new("RGB", size = (w, h), color= color)
    
    plt.imshow(test_image)
    plt.show()    
    print("saving image...")
    date_string = time.strftime("%Y-%m-%d-%H_%M")
    test_image.save("./Images/adversarials/test_image{}.png".format(date_string))
    #img.show()

    

def predictImage(data):
    
    image, labels = data
    image, labels = image.to(device), labels.to(device)
    image.requires_grad = True
    
    output = model.forward(image)
    x_pred = torch.max(output.data, 1)[1][0]   #get an index(class number) of a largest element   
    
    output_probs = F.softmax(output, dim=1)
    x_pred_prob =  torch.max(output_probs.data, 1)[0][0]
    
    print("groundtruth: {} prediction: {} confidence of: {:.2f}%"
          .format(labels.item(), x_pred.item(), x_pred_prob.item()*100))
    
    return image, output, x_pred, x_pred_prob
  
      
def createOneStepAdversarial(image, output, x_pred, x_pred_prob):
           
    y_target = 42
    y_target = torch.tensor([y_target], requires_grad=False)
    print("targetlabel for the attack: {}".format(y_target.item()))
    y_target = y_target.to(device)
    zero_gradients(image)  
    
    loss = torch.nn.CrossEntropyLoss()                     
    loss_cal2 = loss(output, y_target)
    loss_cal2.backward(retain_graph=True)
    
    epsilons = [0, 0.01, 0.15, 0.5 ]
    
    x_grad = torch.sign(image.grad.data)
    
    for epsilon in epsilons:      
        x_adversarial = image.data - epsilon * x_grad
        output_adv = model.forward(Variable(x_adversarial))
        x_adv_pred = torch.max(output_adv.data, 1)[1][0]
        op_adv_probs = F.softmax(output_adv, dim=1)
        adv_pred_prob =  torch.max(op_adv_probs.data, 1)[0][0]
        visualize(image, x_adversarial, 
                  x_grad, epsilon, x_pred, x_adv_pred, 
                  x_pred_prob, adv_pred_prob)

    
def createIterativeAdversarial(image, output, x_pred, x_pred_prob):
    
#    test_image = np.random.randint(256, size=(224, 224, 3), dtype=np.uint8)
#    plt.imshow(test_image)
#    plt.show()

#    image = torch.randn((3, 224, 224), requires_grad = True).to(device)
#    image = image.unsqueeze(0)
#    model.forward(image)
#    print(test_label)
#    print(image.data)
#    loss = torch.nn.CrossEntropyLoss()                     
#    loss_cal2 = loss(output, test_label)
#    print(image.grad.data)
    image_temp = image.clone()
    y_target = 25
    y_target = torch.tensor([y_target], requires_grad=False)
    print("targetlabel for the attack: {}".format(y_target.item()))
    y_target = y_target.to(device)
             
    epsilons = [0.75]
#    epsilons = [0.5]
    num_steps = 20
    alpha = 0.05
    
   # x_adversarial.data = image
    for epsilon in epsilons:
        for i in range(num_steps):
            zero_gradients(image)
            loss = torch.nn.CrossEntropyLoss()                     
            loss_cal2 = loss(output, y_target)
            loss_cal2.backward(retain_graph=True)
            x_grad = alpha * torch.sign(image.grad.data)
            adv_temp = image.data - x_grad
            total_grad = adv_temp - image
            total_grad = torch.clamp(total_grad, -epsilon, epsilon)
            x_adversarial = image + total_grad
            image.data = x_adversarial
    
        output_adv = model.forward(Variable(image))
        x_adv_pred = torch.max(output_adv.data, 1)[1][0]
        op_adv_probs = F.softmax(output_adv, dim=1)
        x_adv_pred_prob =  torch.max(op_adv_probs.data, 1)[0][0]
        visualize(image_temp, image.data, total_grad, epsilon, x_pred, x_adv_pred, 
                  x_pred_prob, x_adv_pred_prob) 
        
    
# FGSM attack code
def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image




# =============================================================================
# def testImages(epsilon, model = model, device = device, 
#                testloader = testloader):
#     
#     print("starting testing...")
#     running_corrects = 0
#     
#     for data in testloader:
#         inputs, labels = data
#         inputs, labels = inputs.to(device), labels.to(device)
#         
#         out = utils.make_grid(inputs)
# 
#         imshow(out, title=[x for x in classes])
#         
#         
#         inputs.requires_grad = True
#         output = model.forward(inputs)
#         pred_label = torch.max(output, 1)[1]
#         print(pred_label)
#         print(labels)
#         if pred_label.item() != labels.item():
#             continue
#         
#         loss = torch.nn.CrossEntropyLoss()
#         loss_cal = loss(output, labels)
# 
#         model.zero_grad()
#         
#         loss_cal.backward(retain_graph=True)
#         
#         data_grad = inputs.grad.data
# 
#         
#         peturbed_inputs = fgsm_attack(inputs, epsilon, data_grad)
#         
#         out = utils.make_grid(peturbed_inputs)
# 
#         imshow(out, title=[x for x in classes])
#         
#         
#         output = model.forward(peturbed_inputs)
#         
#         final_pred = output.max(1, keepdim = True)[1]
#         print(final_pred)
#         
#         if(final_pred.item() == labels.item()):
#             running_corrects += 1    
#     final_acc = running_corrects / float(len(testloader))
#     print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, running_corrects, len(testloader), final_acc))
#         
#     return final_acc
# =============================================================================






def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.squeeze(0)
    print(inp.size())
    inp = inp.detach().to(device).numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated



def visualize(x, x_adv, x_grad, epsilon, clean_pred, adv_pred, clean_prob, adv_prob):
    
    x = x.squeeze(0)     #remove batch dimension # B X C H X W ==> C X H X W
    x = x.mul(torch.FloatTensor(std).to(device).view(3,1,1))
    x = x.add(torch.FloatTensor(mean).to(device).view(3,1,1))
    x = x.detach().to("cpu").numpy()#reverse of normalization op- "unnormalize"
    x = np.transpose( x , (1,2,0))   # C X H X W  ==>   H X W X C
    x = np.clip(x, 0, 1)

    
    x_adv = x_adv.squeeze(0)
    x_adv = x_adv.mul(torch.FloatTensor(std).to(device).view(3,1,1))
    x_adv = x_adv.add(torch.FloatTensor(mean).to(device).view(3,1,1)).to("cpu").numpy()#reverse of normalization op
    x_adv = np.transpose( x_adv , (1,2,0))   # C X H X W  ==>   H X W X C
    x_adv = np.clip(x_adv, 0, 1)
#    print(x_adv.shape)
#    im = Image.fromarray(np.uint8(x_adv*255), "RGB")
#    saving_image = Image.fromarray(x, 'RGB')
#    plt.imshow(im)
#    plt.show()
#    print("saving image...")
#    date_string = time.strftime("%Y-%m-%d-%H_%M")
#    im.save("./Images/adversarials/test_image{}.png".format(date_string))
    
    x_grad = x_grad.squeeze(0).detach().to("cpu").numpy()
    x_grad = np.transpose(x_grad, (1,2,0))
    x_grad = np.clip(x_grad, 0, 1)
    
    figure, ax = plt.subplots(1,2, figsize=(9,4))
    ax[0].imshow(x)
    ax[0].set_title('Clean Example', fontsize=15)
    
    
#    ax[1].imshow(x_grad)
#    ax[1].set_title('Perturbation', fontsize=15)
#    ax[1].set_yticklabels([])
#    ax[1].set_xticklabels([])
#    ax[1].set_xticks([])
#    ax[1].set_yticks([])

    
    ax[1].imshow(x_adv)
    ax[1].set_title('Adversarial Example', fontsize=15)
    
    ax[0].axis('off')
    ax[1].axis('off')

#    ax[0].text(1.1,0.5, "+{}*".format(round(epsilon,3)), size=10, ha="center", 
#             transform=ax[0].transAxes)
    
    ax[0].text(0.5,-0.13, "Prediction: {}\n Probability: {}".format(clean_pred, clean_prob), 
              size=10, ha="center", 
              transform=ax[0].transAxes)
    
#    ax[1].text(1.1,0.5, " = ", size=15, ha="center", transform=ax[1].transAxes)

    ax[1].text(0.5,-0.13, "Prediction: {}\n Probability: {}".format(adv_pred, adv_prob), 
              size=10, ha="center", 
              transform=ax[1].transAxes)
    

    plt.show()


    
#inputs, classes = next(iter(testloader))
#
## Make a grid from batch
#out = utils.make_grid(inputs)
#
#imshow(out, title=[x for x in classes])  


#createImage(random = False, color = "green")

loadModel()
model = model.to(device)


for data in testloader:
    image, output, x_pred, x_pred_prob = predictImage(data)
    createIterativeAdversarial(image, output, x_pred, x_pred_prob)
    
print("finished.")
