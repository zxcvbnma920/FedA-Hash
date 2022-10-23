import sys
import matplotlib
from torch.utils.data import DataLoader, Dataset
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm
from mAP import cal_map
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

from options import args_parser
from update import LocalUpdate
from models import *
from utils import get_dataset_cifar, average_weights, aggregate_att, do_nothing, exp_details, Split_for_each_user, MyCIFAR10, get_data_for_user, load_tinyimagenet_data, Spilt_data_tinyimagenet


class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

path = os.path.abspath(os.path.dirname(__file__))
type_ = sys.getfilesystemencoding()
sys.stdout = Logger('../save/log/new.txt')

if __name__ == '__main__':
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')
    args = args_parser()
    exp_details(args)
    if args.gpu:
        torch.cuda.set_device(args.gpu)
    device = 'cuda' if args.gpu else 'cpu'
    if args.model == 'cnn':
    # Convolutional neural netork
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            global_model = CNNCifar(args=args)
        elif args.dataset == 'tinyimagenet':
            global_model = CNNTinyimagenet(args=args)
    elif args.model == 'AlexNet':
        global_model = AlexNet(args=args)
    elif args.model == 'ResNet':
        if args.dataset == 'mnist':
            global_model = AlexNetMnsit(args=args)
        elif args.dataset == 'cifar10' or 'cifar10-1' or 'cifar10-2':
            global_model = ResNet(args=args)
    else:
        exit('Error: unrecognized model')

    gpus = [0, 1]
    global_model.to(device)
    print(global_model)
    global_model = nn.DataParallel(global_model.cuda(), device_ids=gpus)
    global_weights_init = global_model.state_dict()
    train_loss, train_accuracy = [], []  # train_accuracy is mAP
    print_every = 2
    
    if args.dataset == 'cifar':
        train_dataset, test_dataset, user_groups, user_groups_test = get_dataset_cifar(args)
    elif args.dataset == 'tinyimagenet':
        train_dataset, test_dataset, user_groups, user_groups_test = load_tinyimagenet_data(args)
        
    user_data_list = []
    for idx in range(args.num_users):
        if args.dataset == 'cifar':
            data_for_this_user = Split_for_each_user(args=args, dataset_train=train_dataset, dataset_test=test_dataset,
                                                     user_groups=user_groups, user_groups_test=user_groups_test, idx=idx)
        elif args.dataset == 'tinyimagenet':
            data_for_this_user = Spilt_data_tinyimagenet(args=args, user_groups=user_groups, user_groups_test=user_groups_test, idx=idx)
        user_data_list.append(data_for_this_user)

    # Start of training
    train_loss = []
    result_map_local_before = []
    result_map_local = []
    result_P_local = []
    result_P_H_local = []
    result_map_global = []
    result_P_global = []
    result_P_H_global = []
    local_weights_to_agg = [global_weights_init for i in range(args.num_users)]
    local_weights_to_nextepoch = [global_weights_init for i in range(args.num_users)]
    best_mAP = 0
    best_P = [0 for i in range(8)]
    best_P_H = 0
    best_mAP_before = 0
    best_P_before = [0 for i in range(8)]
    best_P_H_before = 0
    best_mAP_local = 0
    best_P_H_local = 0
    best_P_local = [0 for i in range(8)]
    local_losses = [0 for i in range(args.num_users)]
    for epoch in tqdm(range(args.epochs)):
        print(f'\n | Global Training Round : {epoch+1} |\n')
        train_model = copy.deepcopy(global_model)
        local_model = copy.deepcopy(global_model)
        neg_model = copy.deepcopy(global_model)
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        idxs_users.sort()
        time1 = time.time()
        for idx in idxs_users:
            if idx == user_data_list[idx].idx:
                local_model_train = LocalUpdate(args=args, logger=logger,
                                                trainloader=user_data_list[idx].trainloader,
                                                testloader=user_data_list[idx].testloader,
                                                id=idx)
                if args.algorithm == "fed_multi":
                    if epoch == 0:
                        global_weight, local_weight, loss = local_model_train.update_weights_classify(model=copy.deepcopy(global_model), global_round=epoch)
                    else:
                        local_model.load_state_dict(local_weights_to_nextepoch[idx])
                        global_weight, local_weight, loss = local_model_train.update_weights_mutuallearning(global_model=copy.deepcopy(global_model), global_round=epoch, local_model=copy.deepcopy(local_model))
                elif args.algorithm == "fed_classify":
                    global_weight, local_weight, loss = local_model_train.update_weights_classify(model=copy.deepcopy(global_model), global_round=epoch)
                elif args.algorithm == "moon":
                    if epoch == 0:
                        global_weight, local_weight, loss = local_model_train.update_weights(model=copy.deepcopy(global_model), global_round=epoch)
                    else:
                        neg_model.load_state_dict(local_weights_to_nextepoch[idx])
                        local_model.load_state_dict(global_weights)
                        global_weight, local_weight, loss = local_model_train.update_weights_moon(pos_model=copy.deepcopy(global_model), neg_model=copy.deepcopy(neg_model), global_round=epoch, model=copy.deepcopy(local_model))
                else:
                    global_weight, local_weight, loss = local_model_train.update_weights(model=copy.deepcopy(global_model), global_round=epoch)
                local_weights_to_agg[idx] = copy.deepcopy(global_weight)
                local_weights_to_nextepoch[idx] = copy.deepcopy(local_weight)
                local_losses[idx] = copy.deepcopy(loss)
            else:
                exit('Error: cannot match user id with its data in training period')
        time2 = time.time() - time1
        print('every epoch Run Time sec:{:.4f}s, min:{:.4f}'.format(time2, time2/60))
       
        test_mAP_global = 0
        test_P_global = []
        test_P_H_global = 0
        list_mAP_global, list_P_global, list_P_H_global = [], [], []
        test_model = copy.deepcopy(global_model)
        print('[Test before global aggregation] mAP of each client are computed:')
        for idx in range(args.num_users):
            print('idx = ', idx)
            test_model.load_state_dict(local_weights_to_agg[idx])
            test_model.to(device)
            if idx == user_data_list[idx].idx:
                mAP, P, P_H = cal_map(test_model, user_data_list[idx].trainloader, user_data_list[idx].testloader, args.num_classes)
                list_mAP_global.append(mAP)
                list_P_global.append(P)
                list_P_H_global.append(P_H)
            else:
                exit('Error: cannot match user id with its data in validation period')
        print('local_map_list:', list_mAP_global)
        test_mAP_global = sum(list_mAP_global) / len(list_mAP_global)
        if test_mAP_global > best_mAP:
            best_mAP = test_mAP_global
        result_map_global.append(test_mAP_global)
        list_P_global = np.array(list_P_global)
        test_P_global = list_P_global.mean(axis=0)
        test_P_global = test_P_global.tolist()
        for k in range(8):
            if test_P_global[k] > best_P[k]:
                best_P[k] = test_P_global[k]
        test_P_H_global = sum(list_P_H_global) / len(list_P_H_global)
        if test_P_H_global > best_P_H:
            best_P_H = test_P_H_global
        print('map_mean_list_agg:', result_map_global)
        print('best_global_map:', best_mAP)
        print('best_global_presion@N_mean_list:', best_P)
        print('best_global_presion(hamming dist<=2):', best_P_H)
    
        # update global weights 
        if args.agg_method == 'fedavg':
            global_weights = average_weights(local_weights_to_agg)  # FedAvg
        elif args.agg_method == 'fed_att':
            local_weight = aggregate_att(local_weights_to_agg, global_weights, 1.2, 1, 0.001)  # fed-att
        elif args.agg_method == 'nothing':
            global_weights = do_nothing(local_weights_to_agg)

        # Update global weights
        global_model.load_state_dict(global_weights)
        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)
        print('train_loss', train_loss)
        
    print(train_loss)                                                                                              
    plt.figure()
    plt.title('Training Loss vs Communication rounds')
    plt.plot(range(len(train_loss)), train_loss, color='r')
    plt.ylabel('Training loss')
    plt.xlabel('Communication Rounds')
    plt.savefig('../save/map{}loss.png'.format(best_mAP))
    
    total_time = time.time() - start_time
    print('total Run Time sec:{:.4f}s, min:{:.4f}'.format(total_time, total_time / 60))

