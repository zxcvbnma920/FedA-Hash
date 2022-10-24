import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from mAP import *
import torch.nn.functional as F
from options import args_parser
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
from loss import *
from scipy.linalg import hadamard
import random


class LocalUpdate(object):
    def __init__(self, args, logger, trainloader, testloader, id=-1):
        self.args = args
        self.logger = logger
        self.trainloader = trainloader
        self.testloader = testloader
        self.device = 'cuda' if args.gpu else 'cpu'

        self.classify = nn.CrossEntropyLoss().to(self.device)
        self.id = id
        self.scale = nn.Parameter(torch.zeros(1)).to(self.device)
        self.CSQLoss = CSQLoss(config=args).to(self.device)
        self.DSHLoss = DSHLoss(config=args).to(self.device)
        self.DBDHLoss = DBDHLoss(config=args).to(self.device)
        self.DSHSDLoss = DSHSDLoss(config=args).to(self.device)
        self.QSMIHLoss = QSMIHLoss(config=args).to(self.device)
        self.GreedyHashLoss = GreedyHashLoss(config=args).to(self.device)
        self.DCHLoss = DCHLoss(config=args).to(self.device)
        self.DFHLoss = DFHLoss(config=args).to(self.device)
    
    def update_weights(self, model, global_round):
        model.train()
        epoch_loss = []
        epoch = global_round + 1
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr, weight_decay=self.args.w_d,
                                            momentum=0.9)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=self.args.w_d)
                                             
        elif self.args.optimizer == 'RMSprop':
            optimizer = torch.optim.RMSprop(model.parameters(), lr=self.args.lr, weight_decay=self.args.w_d)
        elif self.args.optimizer == 'adadelta':
            optimizer = torch.optim.Adadelta(model.parameters(), rho=0.9)
        
        lr = self.args.lr * (0.1 ** (epoch // self.args.epoch_lr_decrease))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        if self.args.personal_method == 'fedprox':
            saved_global_model = copy.deepcopy(model)
            global_weight_collector = list(saved_global_model.cuda().parameters())
        
        best_mAP = 0
        best_P = [ 0 for i in range(8)]
        best_P_H = 0
        for iter in range(self.args.local_ep):
                
            all_labels = []
            batch_loss = []
            for batch_idx, (images, labels, ind) in enumerate(self.trainloader):

                label_one_hot = F.one_hot(labels, self.args.num_classes).float().to(self.device)
                images, labels = images.to(self.device), labels.to(self.device)
                labels_numpy = labels.cpu().numpy()
                all_labels = np.concatenate([all_labels, labels_numpy])
                model.zero_grad()
                local_log_probs, local_log_probs_classifer,_ = model(images)
                if self.args.hash_method == 'DSH':
                    hash_loss = self.DSHLoss(local_log_probs, labels)
                elif self.args.hash_method == "CSQ":
                    hash_loss = self.CSQLoss(local_log_probs, labels)
                elif self.args.hash_method == "DBDH":
                    hash_loss = self.DBDHLoss(local_log_probs, label_one_hot, ind, self.args)
                elif self.args.hash_method == "DSHSD":
                    hash_loss = self.DSHSDLoss(local_log_probs, label_one_hot, ind, self.args)
                elif self.args.hash_method == "QSMIH":
                    hash_loss = self.QSMIHLoss(local_log_probs, label_one_hot, ind, self.args)
                elif self.args.hash_method == "GreedyHash":
                    hash_loss = self.GreedyHashLoss(local_log_probs, label_one_hot, ind, self.args)
                elif self.args.hash_method == "DCH":
                    hash_loss = self.DCHLoss(local_log_probs, label_one_hot, ind, self.args)
                elif self.args.hash_method == "DFH":
                    hash_loss = self.DFHLoss(local_log_probs, label_one_hot, ind, self.args)

                if self.args.personal_method == 'fedprox':
                    proximal_term = 0.0
                    for param_index, param in enumerate(model.parameters()):
                        proximal_term += ((self.args.mu / 2) * torch.norm((param - global_weight_collector[param_index])) ** 2)
                    loss = hash_loss + proximal_term
                else:
                    loss = hash_loss

                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            print('|User id : {} | Global Round : {} | Local Epoch : {} | Loss: {:.4f}'.format(self.id, global_round, iter, sum(batch_loss) / len(batch_loss)))
                
        return model.state_dict(), model.state_dict(), sum(epoch_loss) / len(epoch_loss)
       
        
    def update_weights_mutuallearning(self, global_model, global_round, local_model):
       
        # Set mode to train model
        global_model.train()
        local_model.train()
        models = [global_model, local_model]
        epoch = global_round + 1
        epoch_loss = []
        optimizers = []
        for model in models:
            if self.args.optimizer == 'sgd':
                optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                            momentum=0.9)
            elif self.args.optimizer == 'adam':
                optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                             weight_decay=self.args.w_d)
            elif self.args.optimizer == 'RMSprop':
                optimizer = torch.optim.RMSprop(model.parameters(), lr=self.args.lr,
                                            weight_decay=self.args.w_d)
            optimizers.append(optimizer)
          
        lr = self.args.lr * (0.1 ** (epoch // self.args.epoch_lr_decrease))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
        if self.args.personal_method == 'fedprox':
            saved_global_model = copy.deepcopy(models[0])  

        for iter in range(self.args.local_ep):
            batch_loss = []
            all_labels = []
            for batch_idx, (images, labels, ind) in enumerate(self.trainloader):

                label_one_hot = F.one_hot(labels, self.args.num_classes).float().to(self.device)
                images, labels = images.to(self.device), labels.to(self.device)
                labels_numpy = labels.cpu().numpy()
                all_labels = np.concatenate([all_labels, labels_numpy])

                global_log_probs, global_log_probs_classify,_ = models[0](images)
                local_log_probs, local_log_probs_classify,_ = models[1](images)

                global_loss_kl = kl_loss(global_log_probs_classify, local_log_probs_classify)
                global_loss_classify = self.classify(global_log_probs_classify, labels)

                proximal_term = 0.0
                if self.args.personal_method == 'fedprox':
                    for w, w_t in zip(models[0].parameters(), saved_global_model.parameters()):
                        proximal_term += (w - w_t).norm(2)
                    #print("proximal term:", proximal_term)

                if self.args.hash_method == 'DSH':
                    global_loss_hash = self.DSHLoss(global_log_probs, labels)
                elif self.args.hash_method == "CSQ":
                    global_loss_hash = self.CSQLoss(global_log_probs, labels)
                elif self.args.hash_method == "DBDH":
                    global_loss_hash = self.DBDHLoss(global_log_probs, label_one_hot, ind, self.args)
                elif self.args.hash_method == "DSHSD":
                    global_loss_hash = self.DSHSDLoss(global_log_probs, label_one_hot, ind, self.args)
                elif self.args.hash_method == "QSMIH":
                    global_loss_hash = self.QSMIHLoss(global_log_probs, label_one_hot, ind, self.args)
                elif self.args.hash_method == "GreedyHash":
                    global_loss_hash = self.GreedyHashLoss(global_log_probs, label_one_hot, ind, self.args)
                elif self.args.hash_method == "DCH":
                    global_loss_hash = self.DCHLoss(global_log_probs, label_one_hot, ind, self.args)
                elif self.args.hash_method == "DFH":
                    global_loss_hash = self.DFHLoss(global_log_probs, label_one_hot, ind, self.args)

                global_loss = self.args.alpha * global_loss_kl + self.args.beta * global_loss_classify + global_loss_hash + (self.args.mu / 2) * proximal_term
                               
                optimizers[0].zero_grad()
                global_loss.backward(retain_graph=True)
                optimizers[0].step()

                local_loss_kl = kl_loss(local_log_probs_classify, global_log_probs_classify)
                local_loss_classify = self.classify(local_log_probs_classify, labels)

                if self.args.hash_method == 'DSH':
                    local_loss_hash = self.DSHLoss(local_log_probs, labels)
                elif self.args.hash_method == "CSQ":
                    local_loss_hash = self.CSQLoss(local_log_probs, labels)
                elif self.args.hash_method == "DBDH":
                    local_loss_hash = self.DBDHLoss(local_log_probs, label_one_hot, ind, self.args)
                elif self.args.hash_method == "DSHSD":
                    local_loss_hash = self.DSHSDLoss(local_log_probs, label_one_hot, ind, self.args)
                elif self.args.hash_method == "QSMIH":
                    local_loss_hash = self.QSMIHLoss(local_log_probs, label_one_hot, ind, self.args)
                elif self.args.hash_method == "GreedyHash":
                    local_loss_hash = self.GreedyHashLoss(local_log_probs, label_one_hot, ind, self.args)
                elif self.args.hash_method == "DCH":
                    local_loss_hash = self.DCHLoss(local_log_probs, label_one_hot, ind, self.args)
                elif self.args.hash_method == "DFH":
                    local_loss_hash = self.DFHLoss(local_log_probs, label_one_hot, ind, self.args)

                local_loss = self.args.alpha * local_loss_kl + self.args.beta * local_loss_classify + local_loss_hash + (self.args.mu / 2) * proximal_term

                optimizers[1].zero_grad()
                local_loss.backward()
                optimizers[1].step()
                
                loss = [global_loss, local_loss]                        
                batch_loss.append(global_loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            print('|User id : {} | Global Round : {} | Local Epoch : {} | Loss: {:.4f}'.format(self.id, global_round, iter, sum(batch_loss) / len(batch_loss)))
            
        return global_model.state_dict(), local_model.state_dict(), sum(epoch_loss) / len(epoch_loss)
        
        
    def update_weights_classify(self, model, global_round):

        model.train()
        epoch_loss = []
        epoch = global_round+1
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr, weight_decay=self.args.w_d, momentum=0.9)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=self.args.w_d)
        elif self.args.optimizer == 'RMSprop':
            optimizer = torch.optim.RMSprop(model.parameters(), lr=self.args.lr, weight_decay=self.args.w_d)

        if self.args.personal_method == 'fedprox':
            saved_global_model = copy.deepcopy(model)  
            
        best_mAP = 0              
        lr = self.args.lr * (0.1 ** (epoch // self.args.epoch_lr_decrease))        
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr   
            
        for iter in range(self.args.local_ep):
            batch_loss = []
            all_labels = []

            for batch_idx, (images, labels, ind) in enumerate(self.trainloader):

                label_one_hot = F.one_hot(labels, self.args.num_classes).float().to(self.device)
                images, labels = images.to(self.device), labels.to(self.device)
                labels_numpy = labels.cpu().numpy()
                all_labels = np.concatenate([all_labels, labels_numpy])
                model.zero_grad()
                local_log_probs, local_log_probs_classifer,_ = model(images)

                proximal_term = 0.0
                if self.args.personal_method == 'fedprox':
                    for w, w_t in zip(model.parameters(), saved_global_model.parameters()):
                        proximal_term += (w - w_t).norm(2)                 

                if self.args.hash_method == 'DSH':
                    hash_loss = self.DSHLoss(local_log_probs, labels)
                elif self.args.hash_method == "CSQ":
                    hash_loss = self.CSQLoss(local_log_probs, labels)
                elif self.args.hash_method == "DBDH":
                    hash_loss = self.DBDHLoss(local_log_probs, label_one_hot, ind, self.args)
                elif self.args.hash_method == "DSHSD":
                    hash_loss = self.DSHSDLoss(local_log_probs, label_one_hot, ind, self.args)
                elif self.args.hash_method == "QSMIH":
                    hash_loss = self.QSMIHLoss(local_log_probs, label_one_hot, ind, self.args)
                elif self.args.hash_method == "GreedyHash":
                    hash_loss = self.GreedyHashLoss(local_log_probs, label_one_hot, ind, self.args)
                elif self.args.hash_method == "DCH":
                    hash_loss = self.DCHLoss(local_log_probs, label_one_hot, ind, self.args)
                elif self.args.hash_method == "DFH":
                    hash_loss = self.DFHLoss(local_log_probs, label_one_hot, ind, self.args)

                loss = hash_loss + self.args.beta * self.classify(local_log_probs_classifer, labels) + (self.args.mu / 2) * proximal_term

                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            print('|User id : {} | Global Round : {} | Local Epoch : {} | Loss: {:.4f}'.format(self.id, global_round, iter, sum(batch_loss) / len(batch_loss)))
                     
        return model.state_dict(), model.state_dict(), sum(epoch_loss) / len(epoch_loss)

        
        


    
    
