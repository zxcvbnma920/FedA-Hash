import copy
import torch
from PIL import Image
from torchvision import datasets, transforms
from sampling import cifar_iid, cifar_noniid, mnist_iid, mnist_noniid, tinyimagenet_iid, tinyimagenet_noniid
import torch.nn.functional as F
from scipy import linalg
import numpy as np
from torch.utils.data import DataLoader, Dataset
from prefetch_generator import BackgroundGenerator
import pandas as pd
import os
from torchvision.datasets import ImageFolder, DatasetFolder


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)
    def __len__(self):
        return len(self.idxs)
    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label, item

      
class MyCIFAR10(Dataset):  
    def __init__(self, dataset):
        self.dataset = dataset
    def __len__(self):
        L = self.dataset.data.shape[0]
        return L    
    def __getitem__(self, item):
        image, label = self.dataset[item]
        return image, label, item
        
        
class Split_for_each_user():
    def __init__(self, args, dataset_train, dataset_test, user_groups, user_groups_test, idx):
        self.args = args
        self.idx = idx
        self.trainloader, self.testloader = self.train_val_test(dataset_train,
                                                                dataset_test,
                                                                idxs=list(user_groups[idx]),
                                                                idxs_test=list(user_groups_test[idx]))
    def train_val_test(self, dataset_train, dataset_test, idxs, idxs_test):
        trainloader = torch.utils.data.DataLoader(DatasetSplit(dataset_train, idxs),
                                                  batch_size=self.args.local_bs,
                                                  shuffle=True,
                                                  num_workers=16,
                                                  pin_memory=True)
        testloader = torch.utils.data.DataLoader(DatasetSplit(dataset_test, idxs_test),
                                                 batch_size=self.args.local_bs,
                                                 shuffle=False,
                                                 num_workers=16,
                                                 pin_memory=True)
        return trainloader, testloader


def get_dataset_cifar(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """
    if args.dataset == 'cifar':
        data_dir = '../data/cifar/'
        train_transform = transforms.Compose([
            #transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        test_transform = transforms.Compose([
            #transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                         transform=train_transform)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                        transform=test_transform)
        
        if args.iid:
            user_groups, user_groups_test = cifar_iid(train_dataset, test_dataset, args.num_users)
        else:
            user_groups, user_groups_test = cifar_noniid(train_dataset, test_dataset, args.num_users, args.partition)

    elif args.dataset == 'mnist':
        data_dir = '../data/mnist/'
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])
            train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                           transform=apply_transform)
            test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                          transform=apply_transform)
        
        if args.iid:
            user_groups, user_groups_test = mnist_iid(train_dataset, test_dataset, args.num_users)
        else:
            user_groups, user_groups_test = mnist_noniid(train_dataset, test_dataset, args.num_users)

    return train_dataset, test_dataset, user_groups, user_groups_test


class Spilt_data_tinyimagenet():
    """
    Returns train and test datasets.
    """
    def __init__(self, args, user_groups, user_groups_test, idx):
        self.args = args
        self.idx = idx
        self.trainloader, self.testloader = self.train_val_test(idxs=user_groups[idx], idxs_test=user_groups_test[idx])

    def train_val_test(self, idxs, idxs_test):
        data_dir = '../data/tinyimagenet'
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        train_ds = ImageFolder_custom(data_dir+'/train/', dataidxs=idxs, transform=transform)
        test_ds = ImageList(data_dir + '/val/' + 'images', pd.read_csv(data_dir + '/val/val_annotations.csv'), transform=transform, dataidxs=idxs_test)

        train_dl = torch.utils.data.DataLoader(dataset=train_ds, batch_size=self.args.local_bs, shuffle=True, num_workers=32, pin_memory=True)
        test_dl = torch.utils.data.DataLoader(dataset=test_ds, batch_size=self.args.local_bs, shuffle=False, num_workers=32, pin_memory=True)

        return train_dl, test_dl


def load_tinyimagenet_data(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """
    data_dir = '../data/tinyimagenet'
    transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    xray_train_ds = ImageFolder_custom(data_dir+'/train/', transform=transform)
    xray_test_ds = ImageList(data_dir + '/val/' + 'images', pd.read_csv(data_dir + '/val/val_annotations.csv'), transform=transform)

    X_train, y_train = np.array([s[0] for s in xray_train_ds.samples]), np.array([int(s[1]) for s in xray_train_ds.samples])
    X_test, y_test = np.array([img for img in xray_test_ds.imgs]), np.array([label for label in xray_test_ds.labels])
    
    if args.iid :
        user_groups, user_groups_test = tinyimagenet_iid(y_train, y_test, args.num_users)
    else:
        user_groups, user_groups_test = tinyimagenet_noniid(y_train, y_test, args.num_users, args.partition)
    
    return xray_train_ds, xray_test_ds, user_groups, user_groups_test
    
    
class ImageList(object):
  
    def __init__(self, data_path, image_list, transform, dataidxs=None):
        self.transform = transform
        self.dataidxs = dataidxs
        if self.dataidxs is not None:
            self.imgs = np.array([os.path.join(data_path, i) for i in image_list['filename']])[self.dataidxs]
            self.labels = np.array([j for j in image_list['label']])[self.dataidxs]
        else:
            self.imgs = np.array([os.path.join(data_path, i) for i in image_list['filename']])
            self.labels = np.array([j for j in image_list['label']])
       
    def __getitem__(self, index):
        path = self.imgs[index]
        target = self.labels[index]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img, target, index

    def __len__(self):
        return len(self.imgs)
        
        
class ImageFolder_custom(DatasetFolder):
  
    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        imagefolder_obj = ImageFolder(self.root, self.transform, self.target_transform)
        self.loader = imagefolder_obj.loader
        if self.dataidxs is not None:
            self.samples = np.array(imagefolder_obj.samples)[self.dataidxs]
        else:
            self.samples = np.array(imagefolder_obj.samples)

    def __getitem__(self, index):
        path = self.samples[index][0]
        target = self.samples[index][1]
        target = int(target)
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, index

    def __len__(self):
        if self.dataidxs is None:
            return len(self.samples)
        else:
            return len(self.dataidxs)


def do_nothing(w):
    w_next = copy.deepcopy(w[0])
    return w_next


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def aggregate_att(w_clients, w_server, stepsize, metric, dp):
    
    w_next = copy.deepcopy(w_server)
    att, att_mat = {}, {}
    for k in w_server.keys():
        w_next[k] = torch.zeros_like(w_server[k]).cpu()
        att[k] = torch.zeros(len(w_clients)).cpu()
    for k in w_next.keys():
        for i in range(0, len(w_clients)):
            tmp = torch.sub((w_server[k]).cpu(), (w_clients[i][k]).cpu()).abs()
            att[k][i] = torch.norm(tmp).cpu()
    for k in w_next.keys():
        att[k] = F.softmax(att[k], dim=0)
    for k in w_next.keys():
        att_weight = torch.zeros_like(w_server[k]).cpu()
        for i in range(0, len(w_clients)):
            att_weight += torch.mul((w_server[k]).cpu() - (w_clients[i][k]).cpu(), att[k][i])
        w_next[k] = (w_server[k]).cpu() - torch.mul((att_weight).cpu(), stepsize) + torch.mul(
            torch.randn(w_server[k].shape), dp)
    return w_next


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    dataset  : {args.dataset}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    weight_decay  : {args.w_d}')
    print(f'    bit  : {args.bit}')
    print(f'    hash_method  : {args.hash_method}')
    print(f'    Global Rounds   : {args.epochs}')
    print(f'    alpha       : {args.alpha}')
    print(f'    beta       : {args.beta}')
    print(f'    mu       : {args.mu}')
    print(f'    gamma       : {args.gamma}')
    if args.iid:
        print('    iid')
    else:
        print('    non-iid')
    print(f'    algorithm  : {args.algorithm}')
    print(f'    personal_method  : {args.personal_method}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return
