import os
import argparse

import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.model_zoo import load_url

from src.gem_init import init_network, ResNetIR
from pire import pert_each_im

parser = argparse.ArgumentParser(description = "Given a neural features extraction model and an image query, generates a adversarial query.")
parser.add_argument("-T", "--iter", type=int, help="Iterative condition, parameter T in the paper.", default="500")
parser.add_argument("-gpu_id", "--gpu", type=int, help="Using GPU or not, cpu please use -1", default="0")
parser.add_argument("-cnnmodel", "--model", help="Pytorch CNN feature extractor which extracts neural features. Now gem and imagenet-res101 available.", default="gem")
parser.add_argument("-in_dir", "--input_dir", help="Directory for original image queries.", default= "./img_input/")
parser.add_argument("-out_dir", "--output_dir", help="Directory for generated adversarial queries.", default="./img_output/")
parser.add_argument("-perception_op", "--p", help="Whether to use perception optimization, function p in the paper.", default=True)
args = parser.parse_args()



print("Loading network {}...".format(args.model))


# We use pre-trained GeM from http://cmp.felk.cvut.cz/cnnimageretrieval/
if args.model == "gem":
    # download and load GeM model 
    state = load_url('http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/retrievalSfM120k-resnet101-gem-b80fb85.pth',
                    model_dir = './models/')
    net_params = {}
    net_params['architecture'] = state['meta']['architecture']
    net_params['pooling'] = state['meta']['pooling']
    net_params['local_whitening'] = state['meta'].get('local_whitening', False)
    net_params['regional'] = state['meta'].get('regional', False)
    net_params['whitening'] = state['meta'].get('whitening', False)
    net_params['mean'] = state['meta']['mean']
    net_params['std'] = state['meta']['std']
    net_params['pretrained'] = False

    net = init_network(net_params)
    net.load_state_dict(state['state_dict'])
    net.eval()
elif args.model == "imagenet-res101":
    net = models.resnet101(pretrained=True)
    features = list(net.children())[:-1]
    net = ResNetIR(features)

    modules=list(net.children())
    modules[-2][-1] = torch.nn.AdaptiveAvgPool2d((1, 1))
    net=nn.Sequential(*modules)
    for p in net.parameters():
        p.requires_grad = False
    net.eval()
else:    
    print("do not support other networks yet.")





if args.gpu > -1:
    print("Using GPU")
    net.cuda()
    torch_dev = torch.device('cuda:0')
else:
    print("Using CPU")
    net.float()
    torch_dev = torch.device('cpu')
    

print("Generating adversarial image query...")

im_list = []
for item in os.listdir('./img_input/'):
	if item.split('.')[1] == 'jpg':
		im_list.append(item)

for im_name in im_list:
	pert_each_im(im_name, model=net, itr=args.iter, 
		root='./img_input/', save_dir='./img_output/', dev=torch_dev, percep_optim=args.p)


print("Generated adversarial image query have been saved in {}.".format(args.output_dir))