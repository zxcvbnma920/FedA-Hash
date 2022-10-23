#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import random

import numpy as np
from torchvision import datasets, transforms

# def mnist_iid(dataset, num_users):
#     """
#     Sample I.I.D. client data from MNIST dataset
#     :param dataset:
#     :param num_users:
#     :return: dict of image index
#     """
#     num_items = int(len(dataset)/num_users)
#     dict_users, all_idxs = {}, [i for i in range(len(dataset))]
#     for i in range(num_users):
#         dict_users[i] = set(np.random.choice(all_idxs, num_items,
#                                              replace=False))
#         all_idxs = list(set(all_idxs) - dict_users[i])
#     return dict_users


def mnist_noniid_unequal(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset s.t clients
    have unequal amount of data
    :param dataset:
    :param num_users:
    :returns a dict of clients with each clients assigned certain
    number of training imgs
    """
    # 60,000 training imgs --> 50 imgs/shard X 1200 shards
    num_shards, num_imgs = 1200, 50
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # Minimum and maximum shards assigned per client:
    min_shard = 1
    max_shard = 30

    # Divide the shards into random chunks for every client
    # s.t the sum of these chunks = num_shards
    random_shard_size = np.random.randint(min_shard, max_shard+1,
                                          size=num_users)
    random_shard_size = np.around(random_shard_size /
                                  sum(random_shard_size) * num_shards)
    random_shard_size = random_shard_size.astype(int)

    # Assign the shards randomly to each client
    if sum(random_shard_size) > num_shards:

        for i in range(num_users):
            # First assign each client 1 shard to ensure every client has
            # atleast one shard of data
            rand_set = set(np.random.choice(idx_shard, 1, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

        random_shard_size = random_shard_size-1

        # Next, randomly assign the remaining shards
        for i in range(num_users):
            if len(idx_shard) == 0:
                continue
            shard_size = random_shard_size[i]
            if shard_size > len(idx_shard):
                shard_size = len(idx_shard)
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)
    else:

        for i in range(num_users):
            shard_size = random_shard_size[i]
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

        if len(idx_shard) > 0:
            # Add the leftover shards to the client with minimum images:
            shard_size = len(idx_shard)
            # Add the remaining shard to the client with lowest data
            k = min(dict_users, key=lambda x: len(dict_users.get(x)))
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[k] = np.concatenate(
                    (dict_users[k], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

    return dict_users


def cifar_iid(dataset_train, dataset_test, num_users):

    L = np.array(dataset_train.targets)
    L_test = np.array(dataset_test.targets)

    first = True
    for label in range(10):

        index = np.where(L == label)[0]
        '''
        N = index.shape[0]
        perm = np.random.permutation(N)
        index = index[perm]
        '''
        index_test = np.where(L_test == label)[0]
        '''
        N_test = index_test.shape[0]
        perm_test = np.random.permutation(N_test)
        index_test = index_test[perm_test]
        '''
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

def mnist_iid(dataset_train, dataset_test, num_users):

    num_items = int(len(dataset_train)/num_users)
    num_items_test = int(len(dataset_test)/num_users)
    dict_users = {}
    dict_users_test = {}
    all_idxs_train, all_idxs_test = [i for i in range(len(dataset_train))], [j for j in range(len(dataset_test))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs_train, num_items, replace=False))
        dict_users_test[i] = set(np.random.choice(all_idxs_test, num_items_test, replace=False))
        all_idxs_train = list(set(all_idxs_train) - dict_users[i])
        all_idxs_test = list(set(all_idxs_test) - dict_users_test[i])
        
    return dict_users, dict_users_test



def mnist_noniid(dataset_train, dataset_test, num_users):
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 80, 750
    idxs_test = np.arange(len(dataset_test))

    dict_users = {i: np.array([]) for i in range(num_users)}
    dict_users_test = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(len(dataset_train))  
    idxs.astype(int)

    labels = np.array(dataset_train.targets)
    labels_test = np.array(dataset_test.targets)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))  
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]  
    idxs = idxs_labels[0, :]
    idxs2 = idxs_labels[1, :]

    idxs_labels_test = np.vstack((idxs_test, labels_test))  
    idxs_labels_test = idxs_labels_test[:, idxs_labels_test[1, :].argsort()]  
    idxs_test = idxs_labels_test[0, :] 
    idxs2_test = idxs_labels_test[1, :]  

    dict_user_label = {i: np.array([]) for i in range(num_users)}
    dict_user_label_test = {i: np.array([]) for i in range(num_users)}
    # divide and assign
    num_imgs_test = 125

    rand_set = []
    for i in range(num_users):
        rand_set.append(shulie(i, 81-num_users+i, num_users))

    NUm_k = []
    for i in range(num_users):

        for rand in rand_set[i]:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
            dict_user_label[i] = np.concatenate(
                (dict_user_label[i], idxs2[rand * num_imgs:(rand + 1) * num_imgs]),
                axis=0)
            dict_users_test[i] = np.concatenate(
                (dict_users_test[i], idxs_test[rand * num_imgs_test:(rand + 1) * num_imgs_test]), axis=0)
            dict_user_label_test[i] = np.concatenate(
                (dict_user_label_test[i], idxs2_test[rand * num_imgs_test:(rand + 1) * num_imgs_test]),
                axis=0)

        dict_users[i] = dict_users[i].astype(int)
        dict_users_test[i] = dict_users_test[i].astype(int)
        Num_k = []
        for k in range(10):
            num_k = len(np.where(dict_user_label[i] == k)[0])
            Num_k.append(num_k)
        NUm_k.append(Num_k)
    

    return dict_users, dict_users_test
    
'''    
def cifar_noniid(dataset_train, dataset_test, num_users):

    min_size = 0
    K = 10

    N = len(dataset_train.targets)
    np.random.seed(2020)
    dict_users = {}
    dict_users_test = {}
    min_require_size = 4000
    
    while min_size < min_require_size:
        idx_batch = [[] for _ in range(num_users)]
        for k in range(K):
            idx_k = np.where(np.array(dataset_train.targets) == k)[0]
            np.random.shuffle(idx_k)
            proportions_train = np.random.dirichlet(np.repeat(0.5, num_users))
            proportions_train = np.array([p * (len(idx_j) < N / num_users) for p, idx_j in zip(proportions_train, idx_batch)])
            proportions_train = proportions_train / proportions_train.sum()
            proportions_train = (np.cumsum(proportions_train) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions_train))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    for j in range(num_users):
        np.random.shuffle(idx_batch[j])
        dict_users[j] = idx_batch[j]
        
    L_test = np.array(dataset_test.targets)

    first = True
    for label in range(10):

        
        #N = index.shape[0]
        #perm = np.random.permutation(N)
        #index = index[perm]
        
        index_test = np.where(L_test == label)[0]
        
        #N_test = index_test.shape[0]
        #perm_test = np.random.permutation(N_test)
        #index_test = index_test[perm_test]
        
        
        batch_idxs_test = np.array_split(index_test, num_users)
        if first:
            dict_users_test = {i: batch_idxs_test[i] for i in range(num_users)}
            
        else:
            dict_users_test = {i: np.concatenate((dict_users_test[i], batch_idxs_test[i])) for i in range(num_users)}
        first = False

    return dict_users, dict_users_test


'''
'''
def cifar_noniid(dataset_train, dataset_test, num_users):
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs, num_imgs_test = 80, 625, 125
    idxs_test = np.arange(len(dataset_test))

    dict_users = {i: np.array([]) for i in range(num_users)}
    dict_users_test = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(len(dataset_train))  
    idxs.astype(int)

    labels = np.array(dataset_train.targets)
    labels_test = np.array(dataset_test.targets)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))  
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]  
    idxs = idxs_labels[0, :]
    idxs2 = idxs_labels[1, :]

    idxs_labels_test = np.vstack((idxs_test, labels_test))  
    idxs_labels_test = idxs_labels_test[:, idxs_labels_test[1, :].argsort()]  
    idxs_test = idxs_labels_test[0, :] 
    idxs2_test = idxs_labels_test[1, :]  

    dict_user_label = {i: np.array([]) for i in range(num_users)}
    dict_user_label_test = {i: np.array([]) for i in range(num_users)}
    # divide and assign

    rand_set = []
    for i in range(num_users):
        rand_set.append(shulie(i, 81-num_users+i, num_users))

    NUm_k = []
    for i in range(num_users):

        for rand in rand_set[i]:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
            dict_user_label[i] = np.concatenate(
                (dict_user_label[i], idxs2[rand * num_imgs:(rand + 1) * num_imgs]),
                axis=0)
            dict_users_test[i] = np.concatenate(
                (dict_users_test[i], idxs_test[rand * num_imgs_test:(rand + 1) * num_imgs_test]), axis=0)
            dict_user_label_test[i] = np.concatenate(
                (dict_user_label_test[i], idxs2_test[rand * num_imgs_test:(rand + 1) * num_imgs_test]),
                axis=0)

        dict_users[i] = dict_users[i].astype(int)
        dict_users_test[i] = dict_users_test[i].astype(int)
        Num_k = []
        for k in range(10):
            num_k = len(np.where(dict_user_label[i] == k)[0])
            Num_k.append(num_k)
        NUm_k.append(Num_k)
    
    return dict_users, dict_users_test
'''

    
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
        
        #N = index.shape[0]
        #perm = np.random.permutation(N)
        #index = index[perm]
        
        index_test = np.where(L_test == label)[0]
        
        #N_test = index_test.shape[0]
        #perm_test = np.random.permutation(N_test)
        #index_test = index_test[perm_test]
        
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

'''
def tinyimagenet_noniid(y_train, y_test, num_users):
    
    min_size = 0
    K = 200

    N = y_train.shape[0]
    np.random.seed(2020)
    dict_users = {}
    dict_users_test = {}
    min_require_size = 9800
    
    while min_size < min_require_size:
        idx_batch = [[] for _ in range(num_users)]
        for k in range(K):
            idx_k = np.where(y_train == k)[0]
            np.random.shuffle(idx_k)
            proportions_train = np.random.dirichlet(np.repeat(0.5, num_users))
            proportions_train = np.array([p * (len(idx_j) < N / num_users) for p, idx_j in zip(proportions_train, idx_batch)])
            proportions_train = proportions_train / proportions_train.sum()
            proportions_train = (np.cumsum(proportions_train) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions_train))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    for j in range(num_users):
        np.random.shuffle(idx_batch[j])
        dict_users[j] = idx_batch[j]
        
    L_test = y_test
    first = True
    for label in range(200):

        
        #N = index.shape[0]
        #perm = np.random.permutation(N)
        #index = index[perm]
        
        index_test = np.where(L_test == label)[0]
        
        #N_test = index_test.shape[0]
        #perm_test = np.random.permutation(N_test)
        #index_test = index_test[perm_test]
        
        
        batch_idxs_test = np.array_split(index_test, num_users)
        if first:
            dict_users_test = {i: batch_idxs_test[i] for i in range(num_users)}
            
        else:
            dict_users_test = {i: np.concatenate((dict_users_test[i], batch_idxs_test[i])) for i in range(num_users)}
        first = False

    return dict_users, dict_users_test
    
'''
'''
def tinyimagenet_noniid(y_train, y_test, num_users):
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs, num_imgs_test = 1000, 100, 10
    idxs_test = np.arange(y_test.shape[0])
    dict_users = {i: np.array([]) for i in range(num_users)}
    dict_users_test = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(y_train.shape[0])  
    idxs.astype(int)

    labels = y_train
    labels_test = y_test

    # sort labels
    idxs_labels = np.vstack((idxs, labels))  
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]  
    idxs = idxs_labels[0, :]
    idxs2 = idxs_labels[1, :]

    idxs_labels_test = np.vstack((idxs_test, labels_test))  
    idxs_labels_test = idxs_labels_test[:, idxs_labels_test[1, :].argsort()]  
    idxs_test = idxs_labels_test[0, :] 
    idxs2_test = idxs_labels_test[1, :]  

    dict_user_label = {i: np.array([]) for i in range(num_users)}
    dict_user_label_test = {i: np.array([]) for i in range(num_users)}
    # divide and assign

    rand_set = []
    for i in range(num_users):
        rand_set.append(shulie(i, 1251-num_users+i, num_users))

    NUm_k = []
    for i in range(num_users):

        for rand in rand_set[i]:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
            dict_user_label[i] = np.concatenate(
                (dict_user_label[i], idxs2[rand * num_imgs:(rand + 1) * num_imgs]),
                axis=0)
            dict_users_test[i] = np.concatenate(
                (dict_users_test[i], idxs_test[rand * num_imgs_test:(rand + 1) * num_imgs_test]), axis=0)
            dict_user_label_test[i] = np.concatenate(
                (dict_user_label_test[i], idxs2_test[rand * num_imgs_test:(rand + 1) * num_imgs_test]),
                axis=0)
        
        dict_users[i] = dict_users[i].astype(int)
        dict_users_test[i] = dict_users_test[i].astype(int)
        Num_k = []
        for k in range(200):
            num_k = len(np.where(dict_user_label[i] == k)[0])
            Num_k.append(num_k)
        NUm_k.append(Num_k)
    import pdb; pdb.set_trace()
    return dict_users, dict_users_test
'''

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
