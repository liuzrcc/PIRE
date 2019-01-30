import argparse
import os
import numpy as np
import random
from PIL import Image

import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torch.autograd.gradcheck import zero_gradients
from torch.nn.parameter import Parameter


from tqdm import tqdm

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
loader = transforms.Compose([transforms.ToTensor(), normalize])

def image_loader(image_name, dev):
    """load image, returns tensor"""
    image = Image.open(image_name)
    image = loader(image).float()
    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
    return image.to(dev)  #assumes that you're using GPU

def proj_lp(v, xi, p):

    # Project on the lp ball centered at 0 and of radius xi

    if p ==np.inf:
            v = torch.clamp(v,-xi,xi)
    else:
        v = v * min(1, xi/(torch.norm(v,p)+0.00001))
    return v


def data_input_init_sz(xi, h, w, dev):
    mean = [ 0.485, 0.456, 0.406 ]
    std = [ 0.229, 0.224, 0.225 ]
    tf = transforms.Compose([
    transforms.Resize((h, w)),
    transforms.ToTensor(),
    transforms.Normalize(mean = mean,
                         std = std)])
    
    v = (torch.rand(1,3,h,w).to(dev)-0.5)*2*xi
    return (mean,std,tf,v)

def enlarge_to_pixel(new_v, times):
    res = (torch.ceil(torch.abs(new_v) /  0.00390625)  * (torch.sign(new_v))) * 0.004 * times
    return res

def pert_each_im(im_name, model, itr, root, save_dir, dev, percep_optim):
    
    mean = [ 0.485, 0.456, 0.406 ]
    std = [ 0.229, 0.224, 0.225 ]

    image = image_loader(root + im_name, dev=dev)
    image.to(dev)
    h = image.size()[2]
    w = image.size()[3]

    for param in model.parameters():
        param.requires_grad = False

    p=np.inf
    xi=10/255.0

    mean, std,tf,init_v = data_input_init_sz(xi, h, w, dev=dev)

    v = torch.autograd.Variable(init_v.to(dev),requires_grad=True)
    loss_fn = torch.nn.MSELoss(reduction='sum')

    size = model(torch.zeros(1, 3, h, w).to(dev)).size()

    learning_rate = 1e-4
    optimizer = torch.optim.Adam([v], lr=learning_rate)

    image = image_loader(root + im_name, dev = dev)
    gem_out = model(image)
    loss_track = []

    for t in tqdm(range(itr)):    

        # Forward pass: compute predicted y by passing x to the model.
        y_pred = model(image + v)

        # Compute and print loss.
        loss =  -1 * loss_fn(y_pred, gem_out)

#         print(t, loss.item())
        loss_track.append(loss.item())
#         loss =  -1 * torch.sum(y_pred)
#         print(t, loss.data.cpu())
        optimizer.zero_grad()

        v.data = proj_lp(v.data, xi, p)

        loss.backward(retain_graph=True)
        optimizer.step()

    v.data = proj_lp(v.data, xi, p)

    if percep_optim == True:
        large_v = enlarge_to_pixel(v.data, 8)
        modified = image + large_v        
    else:
        modified = image + (10 * v.data)
    
    path = save_dir + im_name
    torchvision.utils.save_image(modified, path, normalize=True)