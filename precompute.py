from utils import log
import torch
import torch.nn as nn
from torchsummary import summary
import time
import csv
import numpy as np

from utils.test_utils import get_measures
import os

from sklearn.linear_model import LogisticRegressionCV
from torch.autograd import Variable
from utils.mahalanobis_lib import sample_estimator
from utils.data_loader import get_dataloader_in, get_dataloader_out, cifar_out_datasets, imagenet_out_datasets
from utils.model_loader import get_model

from utils.cal_score import *
from argparser import *

import torchvision
from torchvision import transforms
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import laplace
import torch.nn.functional as F
from tqdm import tqdm
import pickle



def get_class_mean(args, model, num_classes, train_loader):
    if args.in_dataset == "CIFAR-10" or args.in_dataset == "CIFAR-100":
        temp_x = torch.rand(2,3,32,32).cuda()
    else:
        temp_x = torch.rand(2,3,224,224).cuda()
    temp_x = Variable(temp_x)
    temp_list = model.feature_list(temp_x)[1]
    num_output = len(temp_list)

    feature_list = np.empty(num_output)
    count = 0
    for out in temp_list:
        feature_list[count] = out.size(1)
        count += 1
    class_mean, _ = sample_estimator(model, num_classes, feature_list, train_loader)

    file_folder = f'checkpoints/feature/{args.name}/{args.in_dataset}'
    if not os.path.isdir(file_folder):
        os.makedirs(file_folder)

    class_mean = class_mean[-1]

    np.save(f"{file_folder}/{args.model}_class_mean.npy", class_mean)


kwargs = {'num_workers': 2, 'pin_memory': True}
imagesize = 32
transform_test = transforms.Compose([
    transforms.Resize((imagesize, imagesize)),
    transforms.CenterCrop(imagesize),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test_largescale = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def main(args):
    if args.in_dataset == "CIFAR-10":
        data_path = '/data/cifar10'
        # Data loading code
        trainset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=False, transform=transform_test)
        train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch, shuffle=False, **kwargs)

        num_classes = 10

    elif args.in_dataset == "CIFAR-100":
        data_path = '/data/cifar100'
        # Data loading code
        trainset = torchvision.datasets.CIFAR100(root=data_path, train=True, download=False, transform=transform_test)
        train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch, shuffle=False, **kwargs)

        num_classes = 100

       
    elif args.in_dataset == "imagenet":
        root = '/data/ilsvrc2012'
        # Data loading code
        trainset = torchvision.datasets.ImageFolder(os.path.join(root, 'train'), transform_test_largescale)
        train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch, shuffle=False, **kwargs)

        num_classes = 1000

    args.num_classes = num_classes

    load_ckpt = False
    if args.model_path != None:
        load_ckpt = True

    
    model = get_model(args, num_classes, load_ckpt=load_ckpt)
    model.eval()

    get_class_mean(args, model, num_classes, train_dataloader)


    




if __name__ == "__main__":
    parser = get_argparser()

    args = parser.parse_args()

    main(args)
