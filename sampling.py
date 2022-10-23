#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import random

import numpy as np
from torchvision import datasets, transforms

def cifar_iid(dataset_train, dataset_test, num_users):
    L = np.array(dataset_train.targets)
    L_test = np.array(dataset_test.targets)
    first = True
    for label in range(10):
        index = np.where(L == label)[0]
        index_test = np.where(L_test == label)[0]
        batch_idxs = np.array_split(index, num_users)
        batch_idxs_test = np.array_split(index_test, num_users)
        if first:
            dict_users = {i: batch_idxs[i] for i in range(num_users)}
            dict_users_test = {i: batch_idxs_test[i] for i in range(num_users)}
            
        else:
            dict_users = {i: np.concatenate((dict_users[i], batch_idxs[i])) for i in range(num_users)}
            dict_users_test = {i: np.concatenate((dict_users_test[i], batch_idxs_test[i])) for i in range(num_users)}
        first = False        
    return dict_users, dict_users_test

 
def cifar_noniid(dataset_train, dataset_test, num_users, partition):
    K = 10
    num = K * partition
    times = [0 for i in range(10)]
    contain = []
    for i in range(num_users):
        current = [i % K]
        times[i % K] += 1
        j = 1
        while (j < num):
            ind = random.randint(0, K - 1)
            if (ind not in current):
                j = j + 1
                current.append(ind)
                times[ind] += 1
        contain.append(current)
    dict_users = {i : np.ndarray(0, dtype=np.int64) for i in range(num_users)}
    dict_users_test = {i : np.ndarray(0, dtype=np.int64) for i in range(num_users)}
    for i in range(K):
        idx_k = np.where(np.array(dataset_train.targets) == i)[0]
        idx_k_test = np.where(np.array(dataset_test.targets) == i)[0]
        np.random.shuffle(idx_k)
        np.random.shuffle(idx_k_test)
        split = np.array_split(idx_k, times[i])
        split_test = np.array_split(idx_k_test, times[i])
        ids = 0
        for j in range(num_users):
            if i in contain[j]:
                dict_users[j] = np.append(dict_users[j],split[ids])
                dict_users_test[j] = np.append(dict_users_test[j], split_test[ids])
                ids += 1
    return dict_users, dict_users_test


def tinyimagenet_iid(y_train, y_test, num_users):
    L = y_train
    L_test = y_test
    first = True
    for label in range(200):
        index = np.where(L == label)[0]
        index_test = np.where(L_test == label)[0]
        batch_idxs = np.array_split(index, num_users)
        batch_idxs_test = np.array_split(index_test, num_users)
        if first:
            dict_users = {i: batch_idxs[i] for i in range(num_users)}
            dict_users_test = {i: batch_idxs_test[i] for i in range(num_users)}
        else:
            dict_users = {i: np.concatenate((dict_users[i], batch_idxs[i])) for i in range(num_users)}
            dict_users_test = {i: np.concatenate((dict_users_test[i], batch_idxs_test[i])) for i in range(num_users)}
        first = False
    return dict_users, dict_users_test


def tinyimagenet_noniid(y_train, y_test, num_users, partition):
    K = 200
    num = K * partition
    times = [0 for i in range(K)]
    contain = []
    for i in range(num_users):
        current = [i%K]
        times[i%K] += 1
        j = 1
        while (j < num):
            ind = random.randint(0,K-1)
            if (ind not in current):
                j = j+1
                current.append(ind)
                times[ind] += 1
        contain.append(current)
    dict_users = {i:np.ndarray(0,dtype=np.int64) for i in range(num_users)}
    dict_users_test = {i:np.ndarray(0,dtype=np.int64) for i in range(num_users)}
    for i in range(K):
        idx_k = np.where(y_train==i)[0]
        idx_k_test = np.where(y_test==i)[0]
        np.random.shuffle(idx_k)
        np.random.shuffle(idx_k_test)
        split = np.array_split(idx_k,times[i])
        split_test = np.array_split(idx_k_test,times[i])
        ids=0
        for j in range(num_users):
            if i in contain[j]:
                dict_users[j]=np.append(dict_users[j],split[ids])
                dict_users_test[j]=np.append(dict_users_test[j],split_test[ids])
                ids+=1
    return dict_users, dict_users_test
                
    
def shulie(first, end, step):
    x = []
    for i in np.arange(first, end, step):
        x.append(i)
    return x

if __name__ == '__main__':
    dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,),
                                                            (0.3081,))
                                   ]))
    num = 100
    d = mnist_noniid(dataset_train, num)
