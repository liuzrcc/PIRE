import os
import argparse
from torch.utils.model_zoo import load_url
from src.gem_init import init_network
from pire import pert_each_im
import torch

parser = argparse.ArgumentParser(description = "Given a neural features extraction model, generate a adversarial query.")
parser.add_argument("-T", "--iter", type=int, help="iterative condition.", default="500")
parser.add_argument("-cuda", "--gpu", type=int, help="use GPU or not", default="0")
parser.add_argument("-m", "--model", help="pytorch neural feature extractor", default="gem")
parser.add_argument("-in_dir", "--input_dir", help="directory for original image query", default= "./img_input/")
parser.add_argument("-out_dir", "--output_dir", help="directory for generated adversarial query", default="./img_output/")
parser.add_argument("-perception_op", "--p", help="whether to use perception optimization function p", default=True)
args = parser.parse_args()



print("Loading network {}...".format(args.model))


# We use pre-trained GeM from http://cmp.felk.cvut.cz/cnnimageretrieval/
if args.model == "gem":
    # download and load GeM model 
    state = load_url('http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/retrievalSfM120k-resnet101-gem-b80fb85.pth',
                    model_dir = './models/')
else:
    print("do not support other networks yet.")

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