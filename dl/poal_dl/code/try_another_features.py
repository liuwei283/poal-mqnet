import numpy as np
from dataset import get_dataset, get_handler
from model_new import get_net
from torchvision import transforms

import torch
import time
import torch.optim as optim

from torch.utils.data import DataLoader
from copy import deepcopy

from tqdm import tqdm
import torch.nn.functional as F
from torch.autograd import Variable


import warnings
import argparse
import sys
import os
import re
import random
import math
import datetime


import arguments
from parameters import *
from utils import *

np.set_printoptions(threshold=np.inf, linewidth=np.inf)

from model_new import ResNet_CSI

# from query_strategies import RandomSampling, EntropySampling, EntropySamplingIDEAL, POAL_PSES, MQ_Net

def standardize(scores):
    scores = torch.tensor(scores)
    std, mean = torch.std_mean(scores, unbiased=False)
    scores = (scores - mean) / std

    print("Testing:")
    print(torch.max(scores))
    print(torch.min(scores))

    # scores = torch.exp(scores)
    return scores, std, mean

def sample_estimator(model, num_classes, label_loader): # feature list contains the length of the feature representation in each layer
    """
    compute sample mean and precision (inverse of covariance)
    return: sample_class_mean: list of class mean
            precision: list of precisions
    """
    import sklearn.covariance # calculate covariance matrix
    
    model.eval()
    group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False) # question: why is the assume center is True

    num_sample_per_class = np.empty(num_classes) # 每一个class的sample number
    num_sample_per_class.fill(0) # 先设置成0
    list_features = []
    for i in range(num_classes):
        list_features.append(0) # to store the sample mean

    layers = ('simclr', 'shift')
    if not isinstance(layers, (list, tuple)):
        layers = [layers]
    kwargs = {layer: True for layer in layers}

    features_in = torch.tensor([]).to(cuda_device)
    for data, target, idx in tqdm(label_loader):
        inputs = data.to(cuda_device)
        _, couts = model(inputs, **kwargs)
        features_in = torch.cat((features_in, couts['simclr'].detach()), 0)

        for d in range(data.size(0)):
            label = target[d]
            if num_sample_per_class[label] == 0:
                list_features[label] = features_in[d].view(1, -1)
            else:
                list_features[label] = torch.cat((list_features[label], features_in[d].view(1, -1)), 0)
            num_sample_per_class[label] += 1

    sample_class_mean = torch.Tensor(num_classes, int(len(list_features[0][0]))).cuda()
    for i in range(num_classes):
        sample_class_mean[i] = torch.mean(list_features[i], 0)
    
    precision = 0
    for i in range(num_classes):
        if  i == 0:
            precision = list_features[i] - sample_class_mean[i]
        else:
            precision = torch.cat((precision, list_features[i] - sample_class_mean[i]), 0)

    group_lasso.fit(precision.cpu().numpy())
    precision = group_lasso.precision_
    precision = torch.from_numpy(precision).float().cuda()

    return sample_class_mean, precision   



def get_Mahalanobis_score(model, test_loader, num_classes, sample_mean, precision,  magnitude = 0.01):
    '''
    Compute the proposed Mahalanobis confidence score on input dataset
    return: Mahalanobis score from layer_index
    '''
    model.eval()
    Mahalanobis = []

    layers = ('simclr', 'shift')
    if not isinstance(layers, (list, tuple)):
        layers = [layers]
    kwargs = {layer: True for layer in layers}
    
    for data, target, idx in test_loader:
        
        data, target = data.to(cuda_device), target.to(cuda_device)
        data, target = Variable(data, requires_grad = True), Variable(target)

        _, couts = model(data, **kwargs)
        out_features = couts['simclr'].detach()

        # compute Mahalanobis score
        gaussian_score = 0
        for i in range(num_classes): # 每一个class的距离
            batch_sample_mean = sample_mean[i]
            zero_f = out_features.data - batch_sample_mean
            term_gau = -0.5*torch.mm(torch.mm(zero_f, precision), zero_f.t()).diag() # 马氏距离越小这个越大
            if i == 0:
                gaussian_score = term_gau.view(-1,1)
            else:
                gaussian_score = torch.cat((gaussian_score, term_gau.view(-1,1)), 1)

        # to do: try calculate similarity directly
        
        # # Input_processing
        # sample_pred = gaussian_score.max(1)[1] # index of the maximum gaussian score - i.e. the predicted class
        # batch_sample_mean = sample_mean.index_select(0, sample_pred)
        # zero_f = out_features - Variable(batch_sample_mean)
        # pure_gau = -0.5*torch.mm(torch.mm(zero_f, Variable(precision)), zero_f.t()).diag() # 每个data instance对他们最近的那个class的马氏距离
        # loss = torch.mean(-pure_gau) # transform the objective output to schale
        # loss.backward()
        
        # gradient =  torch.ge(data.grad.data, 0) # data的每一个
        # gradient = (gradient.float() - 0.5) * 2 # gradient > 0 => gradient = 1, gradient < 0 => gradient = -1
        # gradient.index_copy_(1, torch.LongTensor([0]).cuda(), gradient.index_select(1, torch.LongTensor([0]).to(cuda_device)) / (0.2023))
        # gradient.index_copy_(1, torch.LongTensor([1]).cuda(), gradient.index_select(1, torch.LongTensor([1]).to(cuda_device)) / (0.1994))
        # gradient.index_copy_(1, torch.LongTensor([2]).cuda(), gradient.index_select(1, torch.LongTensor([2]).to(cuda_device)) / (0.2010))
        # tempInputs = torch.add(data.data, -magnitude, gradient)


        # noise_out_features = model(Variable(tempInputs, volatile=True), **kwargs)
        # noise_out_features = noise_out_features['simclr'].detach()

        # noise_gaussian_score = 0
        # for i in range(num_classes):
        #     batch_sample_mean = sample_mean[i]
        #     zero_f = noise_out_features.data - batch_sample_mean
        #     term_gau = -0.5*torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
        #     if i == 0:
        #         noise_gaussian_score = term_gau.view(-1,1)
        #     else:
        #         noise_gaussian_score = torch.cat((noise_gaussian_score, term_gau.view(-1,1)), 1)	

        # max_noise_gaussian_score, _ = torch.max(noise_gaussian_score, dim=1)
        # all_mahalanobis.extend(noise_gaussian_score.cpu().numpy())
        # Mahalanobis.extend(max_noise_gaussian_score.cpu().numpy())

        max_gaussian_score, _ = torch.max(gaussian_score, dim=1)
        Mahalanobis.extend(max_gaussian_score.cpu().numpy())

        # print("The shape of Maha scores and all Maha scores:")
        # Mahalanobis = np.array(Mahalanobis)
        # all_mahalanobis = np.array(all_mahalanobis)
        # print(Mahalanobis.shape, all_mahalanobis.shape)

    return np.array(Mahalanobis)


# parameters
print(torch.cuda.is_available()) # check whether gpu is available
args_input = arguments.get_args() # parse the input arguments
NUM_QUERY = args_input.batch # batch size of each active learning iteration
NUM_INIT_LB = args_input.initseed # number of initial size of labeling pool
NUM_ROUND = int(args_input.quota / args_input.batch) # number of active learning iteration
DATA_NAME = args_input.dataset # name of the datasets
STRATEGY_NAME = args_input.ALstrategy # acquisition function / al strategy
MODEL_NAME = args_input.model # e.g. resnet / dl model name



SEED = args_input.seed # random seed number
os.environ['CUDA_VISIBLE_DEVICES'] = str(args_input.gpu)

torch.set_printoptions(profile='full')
#print(args_input.gpu)
#torch.cuda.set_device(args_input.gpu)

# sys.stdout = Logger(os.path.abspath('') + '/../logfile/' + 'epochsbackbone' + str(args_input.epochs) + '_epochsmqnet' + str(args_input.epochs_mqnet) + '_lrbackbone' + str(args_input.lr) + '_lrmqnet' + str(args_input.lr_mqnet) + '_normal_log.txt')
warnings.filterwarnings('ignore')

args = args_pool[DATA_NAME]

# set seed
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.enabled  = True
torch.backends.cudnn.benchmark= False
torch.backends.cudnn.deterministic = True

# load dataset
X_tr, Y_tr, X_te, Y_te = get_dataset(DATA_NAME) # dataset
# X_tr = X_tr
# Y_tr = Y_tr

# print the size of the dataset

# start experiment
n_pool = len(Y_tr) # number of data instance in the training pool initially
n_test = len(Y_te) # number of data instance in the testing dataset
print('Number of labeled pool: {}'.format(NUM_INIT_LB))
print('Number of unlabeled pool: {}'.format(n_pool - NUM_INIT_LB))
print('Number of testing pool: {}'.format(n_test))
print('Batch size: {}'.format(NUM_QUERY))
print('Quota: {}'.format(NUM_ROUND*NUM_QUERY))
print('AL strategy: {}'.format(STRATEGY_NAME))
print('Dataset: {}'.format(DATA_NAME))
print('Model: {}'.format(MODEL_NAME))
print('\n')

iteration = args_input.iteration

# load network
net = get_net(DATA_NAME, MODEL_NAME, STRATEGY_NAME)
handler = get_handler(DATA_NAME)

#print(net)



while (iteration > 0):
    all_acc = []
    acq_time = []

    maha_scores_eval = np.array([])
    print("Experiment iteration: " + str(iteration))
    iteration = iteration - 1
    start = datetime.datetime.now()
    idxs_lb = np.zeros(n_pool, dtype=bool)
    acc = np.zeros(NUM_ROUND)

    # device
    use_cuda = torch.cuda.is_available
    cuda_device = torch.device("cuda" if use_cuda else "cpu")  

    # prepare the CSI mode for feature extraction
    model_csi = ResNet_CSI(3, args['num_class']).to(cuda_device)
    optim_csi = torch.optim.SGD(model_csi.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    sched_csi = torch.optim.lr_scheduler.MultiStepLR(optim_csi, milestones=[100, 150])
    # train
    print("Loading weights for the csi model...")
    model_path = os.path.join(os.path.abspath('')) + '/../weights/CIFAR10_06_csi.pt'
    if os.path.isfile(model_path):
        model_csi.load_state_dict(torch.load(model_path))
    
    for r in range(NUM_ROUND):
        # set new ID data per round - simulation for AL query round
        unlabeled_idxs = np.arange(n_pool)[~idxs_lb]
        unlabeled_Y = Y_tr[unlabeled_idxs]
        idx_unlabeled_in_data = unlabeled_idxs[np.where(unlabeled_Y >= 0)[0]]
        idx_unlabeled_ood_data = unlabeled_idxs[np.where(unlabeled_Y < 0)[0]]
        new_query_id_idxs = np.random.choice(idx_unlabeled_in_data, 300, replace = False)
        new_query_ood_idxs = np.random.choice(idx_unlabeled_ood_data, 200, replace = False)
        idxs_lb[new_query_id_idxs] = True
        idxs_lb[new_query_ood_idxs] = True
          
        idxs_train = np.arange(n_pool)[idxs_lb]
        X_train_full = X_tr[idxs_train]
        Y_train_full = Y_tr[idxs_train]
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
            
        # train
        print("Starting to train the backbone model")
        n_epochs = args['n_epoch']
        dim = X_tr.shape[1:]
        backbone = net(dim = dim, num_classes = args['num_class']).to(cuda_device)
        optimizer  = optim.Adam(backbone.parameters(), **args['optimizer_args'])

        loader_tr = DataLoader(handler(X_train, Y_train, transform=args['transform_train']),
                            shuffle=True, **args['loader_tr_args'])
        
        for epoch in tqdm(range(n_epochs)):
            backbone.train() # set mode to train
            for batch_idx, (x, y, idxs) in enumerate(loader_tr):
                x, y = x.to(cuda_device), y.to(cuda_device)
                optimizer.zero_grad()
                out, e1 = backbone(x)
                loss = F.cross_entropy(out, y)
                loss.backward()
                optimizer.step()
        
        print("Predicting:")
        loader_te = DataLoader(handler(X_te, Y_te, transform=args['transform']),
                            shuffle=False, **args['loader_te_args'])
        
        backbone.eval()
        P = torch.zeros(len(Y_te), dtype=Y_te.dtype)
        with torch.no_grad():
            for x, y, idxs in tqdm(loader_te):
                x, y = x.to(cuda_device), y.to(cuda_device)
                out, e1 = backbone(x)
                
                pred = out.max(1)[1]
                P[idxs] = pred.cpu() # indices of the maximum values of the predicted probabilities
        acc[r] = 1.0 * (Y_te==P).sum().item() / len(Y_te) # accuracy of the model
        print("testing accuracy: " + str(acc[r]))
        
        # extract features from csi model
        idxs_unlabeled = np.arange(n_pool)[~idxs_lb]
        X_unlabel = X_tr[idxs_unlabeled]
        Y_unlabel = Y_tr[idxs_unlabeled]

        label_loader_without_train = DataLoader(handler(X_train, Y_train, transform=args['transform']),
                            shuffle=False, **args['loader_te_args'])
        unlabed_loader_without_train = DataLoader(handler(X_unlabel, Y_unlabel, transform=args['transform']),
                            shuffle=False, **args['loader_te_args'])
        
        sample_mean, sample_cov = sample_estimator(model_csi, args['num_class'], label_loader_without_train) # 回来的是training data的每一层的mean(class不一样)和shared covariance

        
        Maha_score = get_Mahalanobis_score(model_csi, unlabed_loader_without_train, args['num_class'], sample_mean, sample_cov)

        Maha_score = np.asarray(Maha_score, dtype = np.float32)

        Maha_score = standardize(Maha_score)[0].numpy()

        # Maha_avg_score = np.mean(Maha_score, axis = 1)

        inv_Maha_score = np.max(Maha_score) - Maha_score # centered的M score 相对的

        # check the returned score of id and ood data
        id_maha_score = inv_Maha_score[torch.where(Y_unlabel>=0)[0]]
        ood_maha_score = inv_Maha_score[torch.where(Y_unlabel<0)[0]]

        id_raw_maha_score = Maha_score[torch.where(Y_unlabel>=0)[0]]
        ood_raw_maha_score = Maha_score[torch.where(Y_unlabel<0)[0]]

        # print("Returned id and ood maha score mean:")
        # print(np.mean(id_maha_score))
        # print(np.mean(ood_maha_score))

        print("evaluating")

        # for layer_num in range(Maha_score.shape[1]):
        #     Maha_score[:, layer_num] = np.max(Maha_score[:, layer_num]) - Maha_score[:, layer_num]

        # idxs_id = torch.where(Y_unlabel<0)[0].numpy()
        # idxs_ood = torch.where(Y_unlabel>=0)[0].numpy()
        # maha_scores_id = Maha_score[idxs_id]
        # maha_scores_ood = Maha_score[idxs_ood]

        # num_samples, num_outputs, num_classes = inv_Maha_score.shape
        # print("The number of samples, output layers, classes:")
        # print(num_samples, num_outputs, num_classes)

        # check variance and mean for maximum maha score
        # max_maha_scores_id = np.max(maha_scores_id, axis=2)
        # max_maha_scores_ood = np.max(maha_scores_ood, axis = 2)
        mean_id = np.median(id_maha_score, axis = 0)
        mean_ood = np.median(ood_maha_score, axis = 0)
        # raw_mean_id = np.mean(id_raw_maha_score,axis = 0)
        # raw_mean_ood = np.mean(ood_raw_maha_score, axis = 0)
        var_id = np.var(id_maha_score, axis = 0)
        var_ood = np.var(ood_maha_score, axis = 0)

        print("mean id, mean ood, var, id, var ood in different layers:")
        print(mean_id)
        print(mean_ood)

        print(var_id)
        print(var_ood)
        
        # print(var_id)
        # print(var_ood)

        # mean_id = mean_id.reshape(1, mean_id.shape[0])
        # # var_id = var_id.reshape(1, var_id.shape[0])
        # mean_ood = mean_ood.reshape(1, mean_ood.shape[0])
        # # var_ood = var_ood.reshape(1, var_ood.shape[0])

        cur_eval = np.array([mean_id, mean_ood])

        if len(maha_scores_eval) == 0:
            maha_scores_eval = cur_eval.reshape(1, cur_eval.shape[0])
        else:
            maha_scores_eval = np.concatenate((maha_scores_eval, cur_eval.reshape(1, cur_eval.shape[0])), axis = 0)

        if r == 2:
            print(maha_scores_eval)

    np.save("maha_eval.npy", maha_scores_eval)








        


        

        




        
        

    
