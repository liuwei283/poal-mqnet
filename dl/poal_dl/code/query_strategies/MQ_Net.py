import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from .strategy import Strategy
from copy import deepcopy
import random
import os
# from torchlars import LARS
from utils import GradualWarmupScheduler
from torch.utils.data.sampler import SubsetRandomSampler

import torch.nn as nn
import sys


from model_new import ResNet_CSI, ResNet18_LL, LossNet32, QueryNet
from transform_layers import Rotation, CutPerm, RandomColorGrayLayer, RandomResizedCropLayer, ColorJitterLayer

import time
from tqdm import tqdm

# from AL_method.CSI.ccal_util import get_shift_module, get_simclr_augmentation
# from AL_method.CSI.simclr_CSI import csi_train_epoch


class MQ_Net(Strategy):
    def __init__(self, X, Y, idxs_lb, net, handler, args, args_input):
        super(MQ_Net, self).__init__(X, Y, idxs_lb, net, handler, args)

        # initialize net loss net and csi net for purity and informativeness detection
        self.models = self.get_models() # store the backbone, loss module and csi model
        self.clf = self.models['backbone']
        self.args_input = args_input
        
        # initialize scheduler and optimizer
        self.get_optim_configurations(self.models, args_input)

        # Self-supervised learning for CSI
        self.self_sup_train()


    def query(self, n):
        print("Start quering")
        idxs_label = np.arange(self.n_pool)[self.idxs_lb]
        idxs_unlabel = np.arange(self.n_pool)[~self.idxs_lb]

        unlabel_X = self.X[idxs_unlabel]
        unlabel_Y = self.Y[idxs_unlabel]

        # in-distribution label data
        label_X_full = self.X[idxs_label]
        label_Y_full = self.Y[idxs_label]
        a = list(range(label_X_full.shape[0]))
        b = torch.where(label_Y_full<0)[0].numpy() # idx of out of distribution data
        d = sorted(list(set(a).difference(set(b)))) # sorted idx of in distribution data

        if type(label_X_full) is np.ndarray:
            tmp = deepcopy(label_X_full)
            tmp = torch.from_numpy(tmp)
            label_X = torch.index_select(tmp, 0, torch.tensor(d))
            label_X = label_X.numpy().astype(label_X_full.dtype)
        else:
            label_X = torch.index_select(label_X_full, 0, torch.tensor(d))

        label_Y = torch.index_select(label_Y_full, 0, torch.tensor(d))

        label_loader = DataLoader(self.handler(label_X, label_Y, transform=self.args['transform']), shuffle=False, batch_size=500, num_workers=5)
        label_loader_train_transform = DataLoader(self.handler(label_X, label_Y, transform=self.args['transform_train']), shuffle=False, batch_size=64, num_workers=5)
        unlabel_loader = DataLoader(self.handler(unlabel_X, unlabel_Y, transform=self.args['transform']), shuffle=False, batch_size=500, num_workers=5)

        print("Getting labeled features...")
        features_in = self.get_labeled_features(label_loader)
        
        print("Getting unlabeled features...")
        if self.args_input.mqnet_mode == 'CONF':
            informativeness, features_unlabeled, _, _ = self.get_unlabeled_features(unlabel_loader)
        if self.args_input.mqnet_mode == 'LL':
            informativeness, features_unlabeled, _, _ = self.get_unlabeled_features_LL(unlabel_loader)

        print("Getting CSI scores...")
        purity = self.get_CSI_score(features_in, features_unlabeled)
        assert len(informativeness) == len(purity)

        if 'mqnet' in self.models: # initial round, MQNet is not trained yet
            if self.args_input.mqnet_mode == 'LL':
                informativeness, _, _ = standardize(informativeness)
            purity, _, _ = standardize(purity)
            query_scores = informativeness + purity
        else:
            meta_input = construct_meta_input(informativeness, purity)
            print("Query using MQ-Net...")
            query_scores = self.models['mqnet'](meta_input)

        selected_indices = np.argsort(-query_scores.reshape(-1).detach().cpu().numpy())[:n]
        Q_index = idxs_unlabel[selected_indices]
        self.idxs_lb[Q_index] = True

        # Meta-training MQNet
        self.meta_train(Q_index, label_loader_train_transform)
                                           
        return Q_index
    
    def meta_train(self, Q_index, label_loader_train):

        # initialize MQ-Net
        self.models['mqnet'] = QueryNet(input_size=2, inter_dim=64).to(self.device)
        optim_mqnet = torch.optim.SGD(self.models['mqnet'].parameters(), lr=self.args_input.lr_mqnet)
        sched_mqnet = torch.optim.lr_scheduler.MultiStepLR(optim_mqnet, milestones=[int(self.args_input.epochs_mqnet / 2)])

        self.optimizers['mqnet'] = optim_mqnet
        self.schedulers['mqnet'] = sched_mqnet

        new_unlabel_idxs = np.arange(self.n_pool)[~self.idxs_lb]
        new_unlabel_X = self.X[new_unlabel_idxs]
        new_unlabel_Y = self.Y[new_unlabel_idxs]

        delta_X = self.X[Q_index]
        delta_Y = self.Y[Q_index]

        unlabeled_loader = DataLoader(self.handler(new_unlabel_X, new_unlabel_Y, transform=self.args['transform']), batch_size=500, num_workers=5)
        delta_loader = DataLoader(self.handler(delta_X, delta_Y, transform=self.args['transform_train']), batch_size=32, num_workers=5)

        print("Getting labeled features...")
        features_in = self.get_labeled_features(label_loader_train)

        print("Getting unlabeled features...")
        if self.args_input.mqnet_mode == 'CONF':
            informativeness, features_delta, in_ood_masks, indices = self.get_unlabeled_features(delta_loader)
        elif self.args_input.mqnet_mode == 'LL':
            informativeness, features_delta, in_ood_masks, indices = self.get_unlabeled_features_LL(delta_loader)

        purity = self.get_CSI_score(features_in, features_delta)

        if self.args_input.mqnet_mode == 'CONF':
            meta_input = construct_meta_input(informativeness, purity)
        elif self.args_input.mqnet_mode == 'LL':
            informativeness_U, _, _, _ = self.get_unlabeled_features(unlabeled_loader)
            meta_input = construct_meta_input_with_U(informativeness, purity, informativeness_U)

        # For enhancing training efficiency, generate meta-input & in-ood masks once, and save it into a dictionary
        meta_input_dict = {}
        for i, idx in enumerate(indices):
            meta_input_dict[idx.item()] = [meta_input[i].to(self.device), in_ood_masks[i]]

        # Mini-batch Training
        self.mqnet_train(delta_loader, meta_input_dict)

    def mqnet_train(self, delta_loader, meta_input_dict):
        print('>> Training MQNet...')
        for epoch in tqdm(range(self.args_input.epochs_mqnet), leave=False, total=self.args_input.epochs_mqnet):

            self.models['mqnet'].train()
            self.models['backbone'].eval()

            batch_idx = 0
            while (batch_idx < self.args_input.steps_per_epoch):
                for data in tqdm(delta_loader):
                    self.optimizers['mqnet'].zero_grad()
                    inputs, labels, indexs = data[0].to(self.device), data[1].to(self.device), data[2].to(self.device)

                    # get pred_scores through MQNet
                    meta_inputs = torch.tensor([]).to(self.device)
                    in_ood_masks = torch.tensor([]).type(torch.LongTensor).to(self.device)
                    for idx in indexs:
                        meta_inputs = torch.cat((meta_inputs, meta_input_dict[idx.item()][0].reshape((-1, 2))), 0)
                        in_ood_masks = torch.cat((in_ood_masks, meta_input_dict[idx.item()][1]), 0)

                    pred_scores = self.models['mqnet'](meta_inputs)

                    # get target loss
                    mask_labels = labels*in_ood_masks # make the label of OOD points to 0 (to calculate loss)

                    out, features = self.models['backbone'](inputs)
                    true_loss = self.criterion(out, mask_labels)  # ground truth loss
                    mask_true_loss = true_loss*in_ood_masks # make the true_loss of OOD points to 0

                    loss = LossPredLoss(pred_scores, mask_true_loss.reshape((-1, 1)), margin=1)

                    loss.backward()
                    self.optimizers['mqnet'].step()

                    batch_idx += 1
            self.schedulers['mqnet'].step()
        print('>> Finished.')

    # get features extracted from CSI model
    def get_labeled_features(self, labeled_in_loader):
        self.models['csi'].eval()

        layers = ('simclr', 'shift')
        if not isinstance(layers, (list, tuple)):
            layers = [layers]
        kwargs = {layer: True for layer in layers}

        features_in = torch.tensor([]).to(self.device)
        for data in tqdm(labeled_in_loader):
            inputs = data[0].to(self.device)
            _, couts = self.models['csi'](inputs, **kwargs)
            features_in = torch.cat((features_in, couts['simclr'].detach()), 0)
        return features_in
    

    def get_unlabeled_features_LL(self, unlabeled_loader):
        self.models['csi'].eval()
        layers = ('simclr', 'shift')
        if not isinstance(layers, (list, tuple)):
            layers = [layers]
        kwargs = {layer: True for layer in layers}

        # generate entire unlabeled features set
        features_unlabeled = torch.tensor([]).to(self.device)
        pred_loss = torch.tensor([]).to(self.device)
        in_ood_masks = torch.tensor([]).type(torch.LongTensor).to(self.device)
        indices = torch.tensor([]).type(torch.LongTensor).to(self.device)

        for data in tqdm(unlabeled_loader):
            inputs = data[0].to(self.device)
            labels = data[1].to(self.device)
            index = data[2].to(self.device)

            in_ood_mask = labels.ge(0).type(torch.LongTensor).to(self.device) # if 1 then in, if 0 then ood
            in_ood_masks = torch.cat((in_ood_masks, in_ood_mask.detach()), 0)

            out, couts = self.models['csi'](inputs, **kwargs)
            features_unlabeled = torch.cat((features_unlabeled, couts['simclr'].detach()), 0)

            out, features = self.models['backbone'](inputs)
            pred_l = self.models['module'](features)  # pred_loss = criterion(scores, labels) # ground truth loss
            pred_l = pred_l.view(pred_l.size(0))
            pred_loss = torch.cat((pred_loss, pred_l.detach()), 0)

            indices = torch.cat((indices, index), 0)

        return pred_loss.reshape((-1, 1)), features_unlabeled, in_ood_masks.reshape((-1, 1)), indices
    
    def save_model(self, model_path, model_path_suffix):
        torch.save(self.models['backbone'].state_dict(), model_path)
        module_path = model_path_suffix + +'_module.params'
        mqnet_path = model_path_suffix + '_mqnet.params'
        torch.save(self.models['module'].state_dict(), module_path)
        torch.save(self.models['mqnet'].state_dict(), mqnet_path)
    
    def get_unlabeled_features(self, unlabeled_loader):
        self.models['csi'].eval()
        layers = ('simclr', 'shift')
        if not isinstance(layers, (list, tuple)):
            layers = [layers]
        kwargs = {layer: True for layer in layers}

        # generate entire unlabeled features set
        features_unlabeled = torch.tensor([]).to(self.device)
        conf = torch.tensor([]).to(self.device)
        in_ood_masks = torch.tensor([]).type(torch.LongTensor).to(self.device)
        indices = torch.tensor([]).type(torch.LongTensor).to(self.device)

        f = nn.Softmax(dim=1)
        for data in unlabeled_loader:
            inputs = data[0].to(self.device)
            labels = data[1].to(self.device)
            index = data[2].to(self.device)

            in_ood_mask = labels.ge(0).type(torch.LongTensor).to(self.device)
            in_ood_masks = torch.cat((in_ood_masks, in_ood_mask.detach()), 0)

            out, couts = self.models['csi'](inputs, **kwargs)
            features_unlabeled = torch.cat((features_unlabeled, couts['simclr'].detach()), 0)

            out, features = self.models['backbone'](inputs)
            u, _ = torch.max(f(out.data), 1)
            conf = torch.cat((conf, u), 0)

            indices = torch.cat((indices, index), 0)
        uncertainty = 1-conf

        return uncertainty.reshape((-1, 1)), features_unlabeled, in_ood_masks.reshape((-1, 1)), indices

    def get_CSI_score(self, features_in, features_unlabeled):
        # CSI Score
        sim_scores = torch.tensor([]).to(self.device)
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-08)
        for f_u in features_unlabeled:
            f_u_expand = f_u.reshape((1, -1)).expand((len(features_in), -1))
            sim = cos(f_u_expand, features_in)  # .reshape(-1,1)
            max_sim, _ = torch.max(sim, 0)
            # score = max_sim * torch.norm(f_u)
            sim_scores = torch.cat((sim_scores, max_sim.reshape(1)), 0)

        # similarity = negative distance = nagative OODness
        return sim_scores.type(torch.float32).to(self.device).reshape((-1, 1))
    
    def train(self, X_train = None, Y_train = None):
        # train the backbone and loss module
        # initialize dataloader
        print("Starting to train the model - MQ-Net strategy...")

        idxs_train = np.arange(self.n_pool)[self.idxs_lb] # id of labeled data instances
        X_train_full = self.X[idxs_train]
        Y_train_full = self.Y[idxs_train]

        # find ID samples
        a = list(range(Y_train_full.shape[0]))
        b = torch.where(Y_train_full<0)[0].numpy() # idx of out of distribution data
        d = sorted(list(set(a).difference(set(b)))) # sorted idx of in distribution data

        Y_train = torch.index_select(Y_train_full, 0, torch.tensor(d)) # all training Y values (in distribution)
        if type(X_train_full) is np.ndarray:
            tmp = deepcopy(X_train_full)
            tmp = torch.from_numpy(tmp)
            X_train = torch.index_select(tmp, 0, torch.tensor(d))
            X_train = X_train.numpy().astype(X_train_full.dtype)
        else:
            X_train = torch.index_select(X_train_full, 0, torch.tensor(d))

        ood_sample_num = Y_train_full.shape[0] - Y_train.shape[0] # sum of odd sample number in this iteration

        print("Length of X train and Y train:")
        print(X_train.shape)
        print(Y_train.shape)
    
        loader_tr = DataLoader(self.handler(X_train, Y_train, transform=self.args['transform_train']),
                            shuffle=True, **self.args['loader_tr_args'])
        
        if self.args_input.mqnet_mode == "LL":
            for epoch in tqdm(range(self.args_input.epochs), leave=False, total=self.args_input.epochs):
                self.train_epoch_LL(epoch,  loader_tr)
                self.schedulers['backbone'].step()
                self.schedulers['module'].step()
        elif self.args_input.mqnet_mode == "CONF":
            # for epoch in tqdm(range(self.args_input.epochs), leave=False, total=self.args_input.epochs):
            for epoch in range(self.args_input.epochs):
                self.train_epoch(loader_tr)
                self.schedulers['backbone'].step()
        else:
            sys.exit('sorry, no valid mqnet mode. Goodbye!')

        return ood_sample_num
    
    def train_epoch_LL(self, epoch, dataloaders):
        print("Traing (LL mode)")
        self.models['backbone'].train()
        self.models['module'].train()

        batch_idx = 0
        while (batch_idx < self.args_input.steps_per_epoch):
            for data in dataloaders:
                inputs, labels = data[0].to(self.device), data[1].to(self.device)

                self.optimizers['backbone'].zero_grad()
                self.optimizers['module'].zero_grad()

                # Classification loss for in-distribution
                scores, features = self.models['backbone'](inputs)
                target_loss = self.criterion(scores, labels)
                m_backbone_loss = torch.sum(target_loss) / target_loss.size(0)

                # loss module for predLoss
                if epoch > self.args_input.epoch_loss:
                    # After 120 epochs, stop the gradient from the loss prediction module
                    features[0] = features[0].detach()
                    features[1] = features[1].detach()
                    features[2] = features[2].detach()
                    features[3] = features[3].detach()

                pred_loss = self.models['module'](features)
                pred_loss = pred_loss.view(pred_loss.size(0))
                m_module_loss = LossPredLoss(pred_loss, target_loss, margin=1)

                loss = m_backbone_loss + m_module_loss
                loss.backward()
                self.optimizers['backbone'].step()
                self.optimizers['module'].step()

                batch_idx += 1

    def train_epoch(self, dataloaders):
        print("Traing (CONF mode)")

        self.models['backbone'].train()

        batch_idx = 0
        while(batch_idx < self.args_input.steps_per_epoch):
            for data in tqdm(dataloaders, leave=False, total=len(dataloaders)):
                inputs, labels = data[0].to(self.device), data[1].to(self.device)

                self.optimizers['backbone'].zero_grad()

                scores, features = self.models['backbone'](inputs)

                target_loss = self.criterion(scores, labels)

                m_backbone_loss = torch.sum(target_loss) / target_loss.size(0)

                loss = m_backbone_loss

                loss.backward()

                self.optimizers['backbone'].step()

                batch_idx+=1

    def get_models(self):
        # for CIFAR10 and CIFAR100
        backbone = ResNet18_LL(3, self.args['num_class']).to(self.device)
        loss_module = LossNet32().to(self.device)
        model_csi = ResNet_CSI(3, self.args['num_class']).to(self.device)

        models = {'backbone': backbone, 'module': loss_module, 'csi': model_csi}

        return models

    def get_optim_configurations(self, models, args_input):
        print("lr: {}, momentum: {}, decay: {}".format(args_input.lr, args_input.momentum, args_input.weight_decay))

        criterion = nn.CrossEntropyLoss(reduction='none').to(self.device)

        # for backbone
        # optimizer
        if args_input.optimizer == "SGD":
            optimizer = torch.optim.SGD(models['backbone'].parameters(), args_input.lr, momentum=args_input.momentum,
                                        weight_decay=args_input.weight_decay)
        elif args_input.optimizer == "Adam":
            optimizer = torch.optim.Adam(models['backbone'].parameters(), args_input.lr, weight_decay=args_input.weight_decay)
        else:
            optimizer = torch.optim.__dict__[args_input.optimizer](models['backbone'].parameters(), args_input.lr, momentum=args_input.momentum,
                                                            weight_decay=args_input.weight_decay)
        # scheduler
        if args_input.scheduler == "CosineAnnealingLR":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args_input.epochs, eta_min=args_input.min_lr)
        elif args_input.scheduler == "StepLR":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args_input.step_size, gamma=args_input.gamma)
        elif args_input.scheduler == "MultiStepLR":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args_input.milestone)
        else:
            scheduler = torch.optim.lr_scheduler.__dict__[args_input.scheduler](optimizer)

        # for LL model
        optim_module = torch.optim.SGD(models['module'].parameters(), lr=args_input.lr, momentum=args_input.momentum, weight_decay=args_input.weight_decay)
        sched_module = torch.optim.lr_scheduler.MultiStepLR(optim_module, milestones=args_input.milestone)

        # for CSI model
        optim_csi = torch.optim.SGD(models['csi'].parameters(), lr=args_input.lr, momentum=args_input.momentum, weight_decay=args_input.weight_decay)
        sched_csi = torch.optim.lr_scheduler.MultiStepLR(optim_csi, milestones=args_input.milestone)

        optimizers = {'backbone': optimizer, 'module': optim_module, 'csi': optim_csi}
        schedulers = {'backbone': scheduler, 'module': sched_module, 'csi': sched_csi}

        self.criterion = criterion
        self.optimizers = optimizers
        self.schedulers = schedulers
    
    def self_sup_train(self):
        print("Self-sup training:")
        model_path = '../../weights/'+ str(self.args_input.dataset) + '_csi.pt'
        if os.path.isfile(model_path):
            print('Load pre-trained CSI model, named: {}'.format(model_path))
            self.models['csi'].load_state_dict(torch.load(model_path))


# def csi_train(args, model, criterion, optimizer, scheduler, loader, device, simclr_aug=None, linear=None, linear_optim=None):
#     print('>> Train CSI.')
#     time_start = time.time()

#     for epoch in range(1000):
#         csi_train_epoch(args, epoch, model, criterion, optimizer, scheduler, loader, device, simclr_aug, linear, linear_optim)
#         scheduler.step()
#     print('>> Finished, Elapsed Time: {}'.format(time.time()-time_start))

def LossPredLoss(input, target, margin=1.0, reduction='mean'):
    assert input.shape == input.flip(0).shape

    input = (input - input.flip(0))[:len(input) // 2]
    target = (target - target.flip(0))[:len(target) // 2]
    target = target.detach()

    one = 2 * torch.sign(torch.clamp(target, min=0)) - 1

    if reduction == 'mean':
        loss = torch.sum(torch.clamp(margin - one * input, min=0))
        loss = loss / input.size(0)
    elif reduction == 'none':
        loss = torch.clamp(margin - one * input, min=0)
    else:
        NotImplementedError()

    return loss

def standardize(scores):
    std, mean = torch.std_mean(scores, unbiased=False)
    scores = (scores - mean) / std
    scores = torch.exp(scores)
    return scores, std, mean

def standardize_with_U(scores, scores_U):
    std, mean = torch.std_mean(scores_U, unbiased=False)
    scores = (scores - mean) / std
    scores = torch.exp(scores)
    return scores, std, mean

def construct_meta_input(informativeness, purity):
    informativeness, std, mean = standardize(informativeness)
    print("informativeness mean: {}, std: {}".format(mean, std))

    purity, std, mean = standardize(purity)
    print("purity mean: {}, std: {}".format(mean, std))

    # TODO:
    meta_input = torch.cat((informativeness, purity), 1)
    return meta_input


def construct_meta_input_with_U(informativeness, purity, informativeness_U):
    scores, std, mean = standardize_with_U(informativeness, informativeness_U)
    print("informativeness mean: {}, std: {}".format(mean, std))

    purity, std, mean = standardize(purity)
    print("purity mean: {}, std: {}".format(mean, std))

    # TODO:
    meta_input = torch.cat((informativeness, purity), 1)
    return meta_input
    

            
