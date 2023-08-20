import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from .strategy import Strategy
from copy import deepcopy
import random
import os
from torchlars import LARS
from utils import GradualWarmupScheduler
from torch.utils.data.sampler import SubsetRandomSampler

import torch.nn as nn


from model_new import ResNet_CSI, ResNet18_LL, LossNet32, QueryNet
from transform_layers import Rotation, CutPerm, RandomColorGrayLayer, RandomResizedCropLayer, ColorJitterLayer

import time

from AL_method.CSI.ccal_util import get_shift_module, get_simclr_augmentation
from AL_method.CSI.simclr_CSI import csi_train_epoch


class MQ_Net(Strategy):
    def __init__(self, X, Y, idxs_lb, net, handler, args):
        super(MQ_Net, self).__init__(X, Y, idxs_lb, net, handler, args)

        # initialize net loss net and csi net for purity and informativeness detection
        models = self.get_models()

        self.models = models # store the backbone, loss module and csi model
        self.clf = self.models['backbone']
        
        # initialize scheduler and optimizer
        self.criterion, self.optimizers, self.schedulers = self.get_optim_configurations(models)

        # Self-supervised learning for CSI
        models = self.self_sup_train(models, self.criterion, self.optimizers, self.schedulers)


    def query(self, n):
        idxs_label = np.arange(self.n_pool)[self.idxs_lb]
        idxs_unlabel = np.arange(self.n_pool)[self.idxs_lb]

        unlabel_X = self.X[idxs_unlabel]
        unlabel_Y = self.Y[idxs_unlabel]

        # in-distribution label data
        label_X_full = self.X[idxs_label]
        label_Y_full = self.Y[idxs_label]
        a = list(range(label_X_full.shape[0]))
        b = torch.where(label_Y_full<0)[0].numpy() # idx of out of distribution data
        d = sorted(list(set(a).difference(set(b)))) # sorted idx of in distribution data

        label_X = torch.index_select(label_X_full, 0, torch.tensor(d)) # all training Y values (in distribution)
        label_Y = torch.index_select(label_Y_full, 0, torch.tensor(d))

        label_loader = DataLoader(self.handler(label_X, label_Y, transform=self.args['transform']), shuffle=False, batch_size=500, num_workers=5)
        unlabel_loader = DataLoader(self.handler(unlabel_X, unlabel_Y, transform=self.args['transform']), shuffle=False, batch_size=500, num_workers=5)

        features_in = self.get_labeled_features(label_loader)
        informativeness, features_unlabeled, _, _ = self.get_unlabeled_features_LL(unlabel_loader)

        purity = self.get_CSI_score(features_in, features_unlabeled)
        assert len(informativeness) == len(purity)

        if 'mqnet' in self.models: # initial round, MQNet is not trained yet
            informativeness, _, _ = standardize(informativeness)
            purity, _, _ = standardize(purity)
            query_scores = informativeness + purity
        else:
            meta_input = construct_meta_input(informativeness, purity)
            query_scores = self.models['mqnet'](meta_input)

        selected_indices = np.argsort(-query_scores.reshape(-1).detach().cpu().numpy())[n]
        Q_index = idxs_unlabel[selected_indices]
        self.idxs_lb[Q_index] = True

        # Meta-training MQNet
        self.meta_train(Q_index, label_loader)
                                           
        return Q_index
    
    def meta_train(self, Q_index, label_loader):

        # initialize MQ-Net
        self.models['mqnet'] = QueryNet(input_size=2, inter_dim=64).to(self.device)
        optim_mqnet = torch.optim.SGD(self.models['mqnet'].parameters(), lr=0.001)
        sched_mqnet = torch.optim.lr_scheduler.MultiStepLR(optim_mqnet, milestones=[int(100 / 2)])

        self.optimizers['mqnet'] = optim_mqnet
        self.schedulers['mqnet'] = sched_mqnet

        new_unlabel_idxs = np.arange(self.n_pool)[~self.idxs_lb]
        new_unlabel_X = self.X[new_unlabel_idxs]
        new_unlabel_Y = self.Y[new_unlabel_idxs]

        delta_X = self.X[Q_index]
        delta_Y = self.Y[Q_index]

        unlabeled_loader = DataLoader(self.handler(new_unlabel_X, new_unlabel_Y, transform=self.args['transform']), batch_size=500, num_workers=5)
        delta_loader = DataLoader(self.handler(delta_X, delta_Y, transform=self.args['transform_train']), batch_size=max(1, 32), num_workers=5)

        features_in = self.get_labeled_features(label_loader)
        informativeness, features_delta, in_ood_masks, indices = self.get_unlabeled_features_LL(delta_loader)
        purity = self.get_CSI_score(features_in, features_delta)

        informativeness_U, _, _, _ = self.get_unlabeled_features(unlabeled_loader)

        meta_input = construct_meta_input(informativeness, purity, with_U=True, informativeness_U=informativeness_U)

        # For enhancing training efficiency, generate meta-input & in-ood masks once, and save it into a dictionary
        meta_input_dict = {}
        for i, idx in enumerate(indices):
            meta_input_dict[idx.item()] = [meta_input[i].to(self.device), in_ood_masks[i]]

        # Mini-batch Training
        self.mqnet_train(delta_loader, meta_input_dict)

    def mqnet_train(self, delta_loader, meta_input_dict):
        print('>> Train MQNet.')
        for epoch in range(100):

            self.models['mqnet'].train()
            self.models['backbone'].eval()

            batch_idx = 0
            while (batch_idx < 100):
                for data in delta_loader:
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


    def get_labeled_features(self, labeled_in_loader):
        self.models['csi'].eval()

        layers = ('simclr', 'shift')
        if not isinstance(layers, (list, tuple)):
            layers = [layers]
        kwargs = {layer: True for layer in layers}

        features_in = torch.tensor([]).to(self.device)
        for data in labeled_in_loader:
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

        for data in unlabeled_loader:
            inputs = data[0].to(self.device)
            labels = data[1].to(self.device)
            index = data[2].to(self.device)

            in_ood_mask = labels.le(self.args['num_IN_class'] - 1).type(torch.LongTensor).to(self.device) # if 1 then in, if 0 then ood
            in_ood_masks = torch.cat((in_ood_masks, in_ood_mask.detach()), 0)

            out, couts = self.models['csi'](inputs, **kwargs)
            features_unlabeled = torch.cat((features_unlabeled, couts['simclr'].detach()), 0)

            out, features = self.models['backbone'](inputs)
            pred_l = self.models['module'](features)  # pred_loss = criterion(scores, labels) # ground truth loss
            pred_l = pred_l.view(pred_l.size(0))
            pred_loss = torch.cat((pred_loss, pred_l.detach()), 0)

            indices = torch.cat((indices, index), 0)

        return pred_loss.reshape((-1, 1)), features_unlabeled, in_ood_masks.reshape((-1, 1)), indices
    
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

            in_ood_mask = labels.le(self.args['num_IN_class']-1).type(torch.LongTensor).to(self.device)
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
    
        loader_tr = DataLoader(self.handler(X_train, Y_train, transform=self.args['transform_train']),
                            shuffle=True, **self.args['loader_tr_args'])
        
        # set the training model of the
        for epoch in range(200):
            self.train_epoch_LL(epoch,  loader_tr)
            self.schedulers['backbone'].step()
            self.schedulers['module'].step()

        return ood_sample_num
    
    
    def train_epoch_LL(self, epoch, loader_tr):
        self.models['backbone'].train()
        self.models['module'].train()
        
        for batch_idx, (x, y, idxs) in enumerate(loader_tr):
            inputs, labels = x.to(self.device), y.to(self.device)
            self.optimizers['backbone'].zero_grad()
            self.optimizers['module'].zero_grad()

            scores, features = self.models['backbone'](inputs)
            target_loss = self.criterion(scores, labels)
            m_backbone_loss = torch.sum(target_loss) / target_loss.size[0]

            # loss module for predLoss
            if epoch > 120:
                features[0] = features[0].detach()
                features[1] = features[1].detach()
                features[2] = features[2].detach()
                features[3] = features[3].detach()

            pred_loss = self.models['module'](features)
            pred_loss = pred_loss.view(pred_loss.size[0])

            # target loss: entropy loss
            # pred_loss: loss predicted by the loss module
            m_module_loss = LossPredLoss(pred_loss, target_loss, margin = 1)

            loss = m_backbone_loss + m_module_loss
            self.optimizers['backbone'].step()
            self.optimizers['module'].step()


    def get_models(self):
        backbone = ResNet18_LL(3, self.args['num_class']).to(self.device)
        loss_module = LossNet32().to(self.device)
        model_csi = ResNet_CSI(3, self.args['num_class']).to(self.device)

        models = {'backbone': backbone, 'module': loss_module, 'csi': model_csi}

        return models

    def get_optim_configurations(self, models):

        criterion = nn.CrossEntropyLoss(reduction='none').to(self.device)

        # for backbone (may not needed)
        optimizer = torch.optim.Adam(models['backbone'].parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

        # for LL model
        optim_module = torch.optim.SGD(models['module'].parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        sched_module = torch.optim.lr_scheduler.MultiStepLR(optim_module, milestones=[100, 150])

        # for CSI model
        optimizer_csi = torch.optim.SGD(models['csi'].parameters(), lr=0.1, momentum=0.9, weight_decay=1e-6)
        optim_csi = LARS(optimizer_csi, eps=1e-8, trust_coef=0.001)
        sched_csi = torch.optim.lr_scheduler.CosineAnnealingLR(optim_csi, 1000)
        scheduler_warmup_csi = GradualWarmupScheduler(optim_csi, multiplier=10.0, total_epoch=10, after_scheduler=sched_csi)

        optimizers = {'backbone': optimizer, 'module': optim_module, 'csi': optim_csi}
        schedulers = {'backbone': scheduler, 'module': sched_module, 'csi': scheduler_warmup_csi}

        return criterion, optimizers, schedulers
    
    def self_sup_train(self, models, criterion, optimizers, schedulers):
        linear = models['csi'].linear
        linear_optim = torch.optim.Adam(linear.parameters(), lr=1e-3, betas=(.9, .999), weight_decay=5e-4)
        self.args['shift_trans_type'] = 'rotation'
        self.args['shift_trans'], self.args['K_shift'] = get_shift_module(self.args, eval=True)
        self.args['shift_trans'] = self.args['shift_trans'].to(self.device)

        # if a pre-trained CSI exist, just load it
        model_path = 'weights/'+ str(self.args['dataset'])+'_r0.001'+'_csi.pt'

        if os.path.isfile(model_path):
            print('Load pre-trained CSI model, named: {}'.format(model_path))
            models['csi'].load_state_dict(torch.load(model_path))
        else:
            # contrastive learning csi
            contrastive_loader = DataLoader(self.handler(self.X, self.Y, transform=self.args['transform_train']),
                            shuffle=True, **self.args['loader_tr_args'])
            simclr_aug = get_simclr_augmentation(image_size=(32, 32, 3)).to(self.device)  # for CIFAR10, 100

            # Training CSI
            csi_train(self.args, models['csi'], criterion, optimizers['csi'], schedulers['csi'],
                      contrastive_loader, self.device, simclr_aug, linear, linear_optim)

            # SSL save
            torch.save(models['csi'].state_dict(), model_path)


def csi_train(args, model, criterion, optimizer, scheduler, loader, device, simclr_aug=None, linear=None, linear_optim=None):
    print('>> Train CSI.')
    time_start = time.time()

    for epoch in range(1000):
        csi_train_epoch(args, epoch, model, criterion, optimizer, scheduler, loader, device, simclr_aug, linear, linear_optim)
        scheduler.step()
    print('>> Finished, Elapsed Time: {}'.format(time.time()-time_start))

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

def construct_meta_input(informativeness, purity, with_U = False, informativeness_U = None):
    if with_U:
        informativeness = standardize_with_U(informativeness, informativeness_U)
    else:
        informativeness, std, mean = standardize(informativeness)

    print("informativeness mean: {}, std: {}".format(mean, std))

    purity, std, mean = standardize(purity)
    print("purity mean: {}, std: {}".format(mean, std))

    # TODO:
    meta_input = torch.cat((informativeness, purity), 1)
    return meta_input
    

            
