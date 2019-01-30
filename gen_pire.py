import os
import argparse
from torch.utils.model_zoo import load_url
from src.gem_init import init_network
from pire import better_better_pert_each_im

parser = argparse.ArgumentParser(description = "Given a neural features extraction model, generate a adversarial query.")
parser.add_argument("T", type=int, help="iterative condition.", default="500")
parser.add_argument("GPU", help="use GPU or not")
parser.add_argument("-m", "--model", help="pytorch neural feature extractor", default="gem")
parser.add_argument("-in_dir", "--input_dir", default= "./img_input/")
parser.add_argument("-out_dir", "--output_dir", default="./img_output/")
args = parser.parse_args()

print(args.model)




print("loading network {}...".format(args.model))

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

if args.GPU == True:
    net.cuda()
    net.eval()
else:
    net.eval()

print("generating adversarial image query...")

im_list = []
for item in os.listdir('./img_input/'):
	if item.split('.')[1] == 'jpg':
		im_list.append(item)

for im_name in im_list:
	better_better_pert_each_im(im_name, model=net, itr=args.T, 
		root='./img_input/', 
		save_dir='./img_output')


print("generated adversarial images have been saved... in %s", args.output_dir)