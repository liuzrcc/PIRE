import numpy as np
import math
import scipy.misc

import torch
from torch.utils.model_zoo import load_url
from torch.autograd import Variable
from torchvision import transforms
import torchvision

import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

import torchvision.transforms as transforms
from torch.autograd.gradcheck import zero_gradients
from torch.nn.parameter import Parameter
import random

def init_network(params):

    # parse params with default values
    architecture = params.get('architecture', 'resnet101')
    local_whitening = params.get('local_whitening', False)
    pooling = params.get('pooling', 'gem')
    regional = params.get('regional', False)
    whitening = params.get('whitening', False)
    mean = params.get('mean', [0.485, 0.456, 0.406])
    std = params.get('std', [0.229, 0.224, 0.225])
    pretrained = params.get('pretrained', True)

    # get output dimensionality size
    dim = 2048

    # loading network from torchvision
    if pretrained:
        if architecture not in FEATURES:
            # initialize with network pretrained on imagenet in pytorch
            net_in = getattr(torchvision.models, architecture)(pretrained=True)
        else:
            # initialize with random weights, later on we will fill features with custom pretrained network
            net_in = getattr(torchvision.models, architecture)(pretrained=False)
    else:
        # initialize with random weights
        net_in = getattr(torchvision.models, architecture)(pretrained=False)

    # initialize features
    # take only convolutions for features,
    # always ends with ReLU to make last activations non-negative
    if architecture.startswith('alexnet'):
        features = list(net_in.features.children())[:-1]
    elif architecture.startswith('vgg'):
        features = list(net_in.features.children())[:-1]
    elif architecture.startswith('resnet'):
        features = list(net_in.children())[:-2]
    elif architecture.startswith('densenet'):
        features = list(net_in.features.children())
        features.append(nn.ReLU(inplace=True))
    elif architecture.startswith('squeezenet'):
        features = list(net_in.features.children())
    else:
        raise ValueError('Unsupported or unknown architecture: {}!'.format(architecture))

    # initialize local whitening
    if local_whitening:
        lwhiten = nn.Linear(dim, dim, bias=True)
        # TODO: lwhiten with possible dimensionality reduce

        if pretrained:
            lw = architecture
            if lw in L_WHITENING:
                print(">> {}: for '{}' custom computed local whitening '{}' is used"
                    .format(os.path.basename(__file__), lw, os.path.basename(L_WHITENING[lw])))
                whiten_dir = os.path.join(get_data_root(), 'whiten')
                lwhiten.load_state_dict(model_zoo.load_url(L_WHITENING[lw], model_dir=whiten_dir))
            else:
                print(">> {}: for '{}' there is no local whitening computed, random weights are used"
                    .format(os.path.basename(__file__), lw))

    else:
        lwhiten = None
    
    # initialize pooling
    pool = POOLING['gem']()
    
    # initialize regional pooling
    if regional:
        rpool = pool
        rwhiten = nn.Linear(dim, dim, bias=True)
        # TODO: rwhiten with possible dimensionality reduce

        if pretrained:
            rw = '{}-{}-r'.format(architecture, pooling)
            if rw in R_WHITENING:
                print(">> {}: for '{}' custom computed regional whitening '{}' is used"
                    .format(os.path.basename(__file__), rw, os.path.basename(R_WHITENING[rw])))
                whiten_dir = os.path.join(get_data_root(), 'whiten')
                rwhiten.load_state_dict(model_zoo.load_url(R_WHITENING[rw], model_dir=whiten_dir))
            else:
                print(">> {}: for '{}' there is no regional whitening computed, random weights are used"
                    .format(os.path.basename(__file__), rw))

        pool = Rpool(rpool, rwhiten)

    # initialize whitening
    if whitening:
        whiten = nn.Linear(dim, dim, bias=True)
        # TODO: whiten with possible dimensionality reduce

        if pretrained:
            w = architecture
            if local_whitening:
                w += '-lw'
            w += '-' + pooling
            if regional:
                w += '-r'
            if w in WHITENING:
                print(">> {}: for '{}' custom computed whitening '{}' is used"
                    .format(os.path.basename(__file__), w, os.path.basename(WHITENING[w])))
                whiten_dir = os.path.join(get_data_root(), 'whiten')
                whiten.load_state_dict(model_zoo.load_url(WHITENING[w], model_dir=whiten_dir))
            else:
                print(">> {}: for '{}' there is no whitening computed, random weights are used"
                    .format(os.path.basename(__file__), w))
    else:
        whiten = None

    # create meta information to be stored in the network
    meta = {
        'architecture' : architecture, 
        'local_whitening' : local_whitening, 
        'pooling' : pooling, 
        'regional' : regional, 
        'whitening' : whitening, 
        'mean' : mean, 
        'std' : std,
        'outputdim' : dim,
    }

    # create a generic image retrieval network
    net = ImageRetrievalNet(features, lwhiten, pool, whiten, meta)

    # initialize features with custom pretrained network if needed
    if pretrained and architecture in FEATURES:
        print(">> {}: for '{}' custom pretrained features '{}' are used"
            .format(os.path.basename(__file__), architecture, os.path.basename(FEATURES[architecture])))
        model_dir = os.path.join(get_data_root(), 'networks')
        net.features.load_state_dict(model_zoo.load_url(FEATURES[architecture], model_dir=model_dir))

    return net

def l2n(x, eps=1e-6):
    return x / (torch.norm(x, p=2, dim=1, keepdim=True) + eps).expand_as(x)
  
class L2N(nn.Module):

    def __init__(self, eps=1e-6):
        super(L2N,self).__init__()
        self.eps = eps

    def forward(self, x):
        return l2n(x, eps=self.eps)
        
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'eps=' + str(self.eps) + ')'
    
class ImageRetrievalNet(nn.Module):
    
    def __init__(self, features, lwhiten, pool, whiten, meta):
        super(ImageRetrievalNet, self).__init__()
        self.features = nn.Sequential(*features)
        self.lwhiten = lwhiten
        self.pool = pool
        self.whiten = whiten
        self.norm = L2N()
        self.meta = meta
    
    def forward(self, x):
        # x -> features
        o = self.features(x)

        # TODO: properly test (with pre-l2norm and/or post-l2norm)
        # if lwhiten exist: features -> local whiten
        if self.lwhiten is not None:
            # o = self.norm(o)
            s = o.size()
            o = o.permute(0,2,3,1).contiguous().view(-1, s[1])
            o = self.lwhiten(o)
            o = o.view(s[0],s[2],s[3],self.lwhiten.out_features).permute(0,3,1,2)
            # o = self.norm(o)

        # features -> pool -> norm
        o = self.norm(self.pool(o)).squeeze(-1).squeeze(-1)

        # if whiten exist: pooled features -> whiten -> norm
        if self.whiten is not None:
            o = self.norm(self.whiten(o))

        # permute so that it is Dx1 column vector per image (DxN if many images)
        return o.permute(1,0)

    def __repr__(self):
        tmpstr = super(ImageRetrievalNet, self).__repr__()[:-1]
        tmpstr += self.meta_repr()
        tmpstr = tmpstr + ')'
        return tmpstr

    def meta_repr(self):
        tmpstr = '  (' + 'meta' + '): dict( \n' # + self.meta.__repr__() + '\n'
        tmpstr += '     architecture: {}\n'.format(self.meta['architecture'])
        tmpstr += '     local_whitening: {}\n'.format(self.meta['local_whitening'])
        tmpstr += '     pooling: {}\n'.format(self.meta['pooling'])
        tmpstr += '     regional: {}\n'.format(self.meta['regional'])
        tmpstr += '     whitening: {}\n'.format(self.meta['whitening'])
        tmpstr += '     outputdim: {}\n'.format(self.meta['outputdim'])
        tmpstr += '     mean: {}\n'.format(self.meta['mean'])
        tmpstr += '     std: {}\n'.format(self.meta['std'])
        tmpstr = tmpstr + '  )\n'
        return tmpstr

    
def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)
    # return F.lp_pool2d(F.threshold(x, eps, eps), p, (x.size(-2), x.size(-1))) # alternative
class GeM(nn.Module):

    def __init__(self, p=3, eps=1e-6):
        super(GeM,self).__init__()
        self.p = Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'


def init_network(params):

    # parse params with default values
    architecture = params.get('architecture', 'resnet101')
    local_whitening = params.get('local_whitening', False)
    pooling = params.get('pooling', 'gem')
    regional = params.get('regional', False)
    whitening = params.get('whitening', False)
    mean = params.get('mean', [0.485, 0.456, 0.406])
    std = params.get('std', [0.229, 0.224, 0.225])
    pretrained = params.get('pretrained', True)

    # get output dimensionality size
    dim = 2048

    # loading network from torchvision
    if pretrained:
        if architecture not in FEATURES:
            # initialize with network pretrained on imagenet in pytorch
            net_in = getattr(torchvision.models, architecture)(pretrained=True)
        else:
            # initialize with random weights, later on we will fill features with custom pretrained network
            net_in = getattr(torchvision.models, architecture)(pretrained=False)
    else:
        # initialize with random weights
        net_in = getattr(torchvision.models, architecture)(pretrained=False)

    # initialize features
    # take only convolutions for features,
    # always ends with ReLU to make last activations non-negative
    if architecture.startswith('alexnet'):
        features = list(net_in.features.children())[:-1]
    elif architecture.startswith('vgg'):
        features = list(net_in.features.children())[:-1]
    elif architecture.startswith('resnet'):
        features = list(net_in.children())[:-2]
    elif architecture.startswith('densenet'):
        features = list(net_in.features.children())
        features.append(nn.ReLU(inplace=True))
    elif architecture.startswith('squeezenet'):
        features = list(net_in.features.children())
    else:
        raise ValueError('Unsupported or unknown architecture: {}!'.format(architecture))

    # initialize local whitening
    if local_whitening:
        lwhiten = nn.Linear(dim, dim, bias=True)
        # TODO: lwhiten with possible dimensionality reduce

        if pretrained:
            lw = architecture
            if lw in L_WHITENING:
                print(">> {}: for '{}' custom computed local whitening '{}' is used"
                    .format(os.path.basename(__file__), lw, os.path.basename(L_WHITENING[lw])))
                whiten_dir = os.path.join(get_data_root(), 'whiten')
                lwhiten.load_state_dict(model_zoo.load_url(L_WHITENING[lw], model_dir=whiten_dir))
            else:
                print(">> {}: for '{}' there is no local whitening computed, random weights are used"
                    .format(os.path.basename(__file__), lw))

    else:
        lwhiten = None
    
    # initialize pooling
    pool = POOLING['gem']()
    
    # initialize regional pooling
    if regional:
        rpool = pool
        rwhiten = nn.Linear(dim, dim, bias=True)
        # TODO: rwhiten with possible dimensionality reduce

        if pretrained:
            rw = '{}-{}-r'.format(architecture, pooling)
            if rw in R_WHITENING:
                print(">> {}: for '{}' custom computed regional whitening '{}' is used"
                    .format(os.path.basename(__file__), rw, os.path.basename(R_WHITENING[rw])))
                whiten_dir = os.path.join(get_data_root(), 'whiten')
                rwhiten.load_state_dict(model_zoo.load_url(R_WHITENING[rw], model_dir=whiten_dir))
            else:
                print(">> {}: for '{}' there is no regional whitening computed, random weights are used"
                    .format(os.path.basename(__file__), rw))

        pool = Rpool(rpool, rwhiten)

    # initialize whitening
    if whitening:
        whiten = nn.Linear(dim, dim, bias=True)
        # TODO: whiten with possible dimensionality reduce

        if pretrained:
            w = architecture
            if local_whitening:
                w += '-lw'
            w += '-' + pooling
            if regional:
                w += '-r'
            if w in WHITENING:
                print(">> {}: for '{}' custom computed whitening '{}' is used"
                    .format(os.path.basename(__file__), w, os.path.basename(WHITENING[w])))
                whiten_dir = os.path.join(get_data_root(), 'whiten')
                whiten.load_state_dict(model_zoo.load_url(WHITENING[w], model_dir=whiten_dir))
            else:
                print(">> {}: for '{}' there is no whitening computed, random weights are used"
                    .format(os.path.basename(__file__), w))
    else:
        whiten = None

    # create meta information to be stored in the network
    meta = {
        'architecture' : architecture, 
        'local_whitening' : local_whitening, 
        'pooling' : pooling, 
        'regional' : regional, 
        'whitening' : whitening, 
        'mean' : mean, 
        'std' : std,
        'outputdim' : dim,
    }

    # create a generic image retrieval network
    net = ImageRetrievalNet(features, lwhiten, pool, whiten, meta)

    # initialize features with custom pretrained network if needed
    if pretrained and architecture in FEATURES:
        print(">> {}: for '{}' custom pretrained features '{}' are used"
            .format(os.path.basename(__file__), architecture, os.path.basename(FEATURES[architecture])))
        model_dir = os.path.join(get_data_root(), 'networks')
        net.features.load_state_dict(model_zoo.load_url(FEATURES[architecture], model_dir=model_dir))

    return net

def l2n(x, eps=1e-6):
    return x / (torch.norm(x, p=2, dim=1, keepdim=True) + eps).expand_as(x)
  
class L2N(nn.Module):

    def __init__(self, eps=1e-6):
        super(L2N,self).__init__()
        self.eps = eps

    def forward(self, x):
        return l2n(x, eps=self.eps)
        
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'eps=' + str(self.eps) + ')'
    
class ImageRetrievalNet(nn.Module):
    
    def __init__(self, features, lwhiten, pool, whiten, meta):
        super(ImageRetrievalNet, self).__init__()
        self.features = nn.Sequential(*features)
        self.lwhiten = lwhiten
        self.pool = pool
        self.whiten = whiten
        self.norm = L2N()
        self.meta = meta
    
    def forward(self, x):
        # x -> features
        o = self.features(x)

        # TODO: properly test (with pre-l2norm and/or post-l2norm)
        # if lwhiten exist: features -> local whiten
        if self.lwhiten is not None:
            # o = self.norm(o)
            s = o.size()
            o = o.permute(0,2,3,1).contiguous().view(-1, s[1])
            o = self.lwhiten(o)
            o = o.view(s[0],s[2],s[3],self.lwhiten.out_features).permute(0,3,1,2)
            # o = self.norm(o)

        # features -> pool -> norm
        o = self.norm(self.pool(o)).squeeze(-1).squeeze(-1)

        # if whiten exist: pooled features -> whiten -> norm
        if self.whiten is not None:
            o = self.norm(self.whiten(o))

        # permute so that it is Dx1 column vector per image (DxN if many images)
        return o.permute(1,0)

    def __repr__(self):
        tmpstr = super(ImageRetrievalNet, self).__repr__()[:-1]
        tmpstr += self.meta_repr()
        tmpstr = tmpstr + ')'
        return tmpstr

    def meta_repr(self):
        tmpstr = '  (' + 'meta' + '): dict( \n' # + self.meta.__repr__() + '\n'
        tmpstr += '     architecture: {}\n'.format(self.meta['architecture'])
        tmpstr += '     local_whitening: {}\n'.format(self.meta['local_whitening'])
        tmpstr += '     pooling: {}\n'.format(self.meta['pooling'])
        tmpstr += '     regional: {}\n'.format(self.meta['regional'])
        tmpstr += '     whitening: {}\n'.format(self.meta['whitening'])
        tmpstr += '     outputdim: {}\n'.format(self.meta['outputdim'])
        tmpstr += '     mean: {}\n'.format(self.meta['mean'])
        tmpstr += '     std: {}\n'.format(self.meta['std'])
        tmpstr = tmpstr + '  )\n'
        return tmpstr

    
def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)
    # return F.lp_pool2d(F.threshold(x, eps, eps), p, (x.size(-2), x.size(-1))) # alternative
class GeM(nn.Module):

    def __init__(self, p=3, eps=1e-6):
        super(GeM,self).__init__()
        self.p = Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'

pool = GeM
POOLING = {
    'gem'  : GeM}


class ResNetIR(nn.Module):
    
    def __init__(self, features):
        super(ResNetIR, self).__init__()
        self.features = nn.Sequential(*features)
        self.norm = L2N()
    
    def forward(self, x):
        # x -> features
        o = self.features(x)


        # features -> pool -> norm
        o = self.norm(o).squeeze(-1).squeeze(-1)


        # permute so that it is Dx1 column vector per image (DxN if many images)
        return o.permute(1,0)