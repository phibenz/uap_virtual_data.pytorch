from __future__ import division
import os
import numpy as np
import glob
import torch
import random
# import cv2
from torch.utils.data import Dataset

import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.transforms.functional as tfunc

from config.config import IMAGENET_PATH, DATASET_BASE_PATH
from config.config import COCO_2017_TRAIN_IMGS, COCO_2017_VAL_IMGS, COCO_2017_TRAIN_ANN, COCO_2017_VAL_ANN, VOC_2012_ROOT, PLACES365_ROOT
from dataset_utils.voc0712 import VOCDetection


def get_data_specs(pretrained_dataset):
    if pretrained_dataset == "imagenet":
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        num_classes = 1000
        input_size = 224
        # input_size = 299 # inception_v3
        num_channels = 3
    elif pretrained_dataset == "cifar10":
        mean = [0., 0., 0.]
        std = [1., 1., 1.]
        num_classes = 10
        input_size = 32
        num_channels = 3
    elif pretrained_dataset == "cifar100":
        mean = [0., 0., 0.]
        std = [1., 1., 1.]
        num_classes = 100
        input_size = 32
        num_channels = 3
    else:
        raise ValueError
    return num_classes, (mean, std), input_size, num_channels


def get_data(dataset, pretrained_dataset):

    num_classes, (mean, std), input_size, num_channels = get_data_specs(pretrained_dataset)

    if dataset == 'cifar10':
        train_transform = transforms.Compose(
                [transforms.RandomHorizontalFlip(),
                 transforms.RandomCrop(input_size, padding=4),
                 transforms.ToTensor(),
                 transforms.Normalize(mean, std)])

        test_transform = transforms.Compose(
                [transforms.ToTensor(),
                transforms.Normalize(mean, std)])

        train_data = dset.CIFAR10(DATASET_BASE_PATH, train=True, transform=train_transform, download=True)
        test_data = dset.CIFAR10(DATASET_BASE_PATH, train=False, transform=test_transform, download=True)
    
    elif dataset == 'cifar100':
        train_transform = transforms.Compose(
                [transforms.RandomHorizontalFlip(),
                 transforms.RandomCrop(input_size, padding=4),
                 transforms.ToTensor(),
                 transforms.Normalize(mean, std)])

        test_transform = transforms.Compose(
                [transforms.ToTensor(),
                transforms.Normalize(mean, std)])

        train_data = dset.CIFAR100(DATASET_BASE_PATH, train=True, transform=train_transform, download=True)
        test_data = dset.CIFAR100(DATASET_BASE_PATH, train=False, transform=test_transform, download=True)
    
    elif dataset == "imagenet":
        traindir = os.path.join(IMAGENET_PATH, 'train')
        valdir = os.path.join(IMAGENET_PATH, 'val')

        train_transform = transforms.Compose([
                transforms.Resize(256),
                # transforms.Resize(299), # inception_v3
                transforms.RandomCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])

        test_transform = transforms.Compose([
                transforms.Resize(256),
                # transforms.Resize(299), # inception_v3
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])

        train_data = dset.ImageFolder(root=traindir, transform=train_transform)
        test_data = dset.ImageFolder(root=valdir, transform=test_transform)
    
    elif dataset == "coco":
        train_transform = transforms.Compose([
                transforms.Resize(int(input_size * 1.143)),
                transforms.RandomCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])

        test_transform = transforms.Compose([
                transforms.Resize(int(input_size * 1.143)),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])

        train_data = dset.CocoDetection(root=COCO_2017_TRAIN_IMGS,
                                        annFile=COCO_2017_TRAIN_ANN,
                                        transform=train_transform)
        test_data = dset.CocoDetection(root=COCO_2017_VAL_IMGS,
                                        annFile=COCO_2017_VAL_ANN,
                                        transform=test_transform)
    
    elif dataset == "voc":
        train_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(int(input_size * 1.143)),
                transforms.RandomCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])

        test_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(int(input_size * 1.143)),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])

        train_data = VOCDetection(root=VOC_2012_ROOT,
                                year="2012",
                                image_set='train',
                                transform=train_transform)
        test_data = VOCDetection(root=VOC_2012_ROOT,
                                year="2012",
                                image_set='val',
                                transform=test_transform)
    
    elif dataset == "places365":
        traindir = os.path.join(PLACES365_ROOT, "train")
        testdir = os.path.join(PLACES365_ROOT, "train")
        # Places365 downloaded as 224x224 images

        train_transform = transforms.Compose([
                transforms.Resize(input_size), # Places images downloaded as 224
                transforms.RandomCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])

        test_transform = transforms.Compose([
                transforms.Resize(input_size),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])
        
        train_data = dset.ImageFolder(root=traindir, transform=train_transform)
        test_data = dset.ImageFolder(root=testdir, transform=test_transform)
    
    return train_data, test_data
