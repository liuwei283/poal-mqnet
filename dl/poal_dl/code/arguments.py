import warnings
import argparse
import sys
import os
import re

# import yaml
from ast import literal_eval
import copy

	
def get_args():
	parser = argparse.ArgumentParser(description='POAL') # descriptions of the software

	#basic arguments
	parser.add_argument('--ALstrategy', '-a', default='EntropySampling', type=str, help='name of active learning strategies') # acquisition functions / AL strategies
	parser.add_argument('--quota', '-q', default=1000, type=int, help='quota of active learning') # human labeling quota
	parser.add_argument('--batch', '-b', default=128, type=int, help='batch size in one active learning iteration')
	parser.add_argument('--dataset', '-d', default='CIFAR10_06', type=str, help='dataset name')
	parser.add_argument('--iteration', '-t', default=3, type=int, help='time of repeat the experiment')
	parser.add_argument('--data_path', type=str, default='./../data', help='Path to where the data is')
	parser.add_argument('--out_path', type=str, default='./../results', help='Path to where the output log will be')
	parser.add_argument('--log_name', type=str, default='test.log', help='middle outputs')
	#parser.add_argument('--help', '-h', default=False, action='store_true', help='verbose')
	parser.add_argument('--cuda', action='store_true', help='If training is to be done on a GPU')
	parser.add_argument('--model', '-m', default='ResNet18', type=str, help='dataset name')
	parser.add_argument('--initseed', '-s', default = 500, type = int, help = 'Initial pool of labeled data') # the number of data in the labled pool
	parser.add_argument('--gpu', '-g', default = 4, type = int, help = 'which gpu') # the number of GPU
	parser.add_argument('--seed', default=4666, type=int, help='random seed')
	parser.add_argument('--num_class', default=10, type=int, help='number of image classes (default: 10 for CIFAR10)')

	### parameters for mqnet (baselines)
	# optimizers and schedulers
	parser.add_argument('--optimizer', default="SGD", help='optimizer to use, e.g. SGD, Adam')
	parser.add_argument("--scheduler", default="MultiStepLR", type=str, help="Learning rate scheduler") #CosineAnnealingLR, StepLR, MultiStepLR
	parser.add_argument('--lr', type=float, default=0.1, help='learning rate for updating network parameters')
	parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum (default: 0.9)')
	parser.add_argument('-wd', '--weight_decay', default=5e-4, type=float, metavar='W', help='weight decay (default: 5e-4)', dest='weight_decay')
	parser.add_argument('--epochs', default=200, type=int, help='number of total epochs to run')
	parser.add_argument('--epochs-csi', default=1000, type=int, help='number of epochs for training CSI')
	parser.add_argument('--min_lr', type=float, default=1e-4, help='minimum learning rate for CosineAnnealingLR')
	parser.add_argument("--gamma", type=float, default=.5, help="Gamma value for StepLR")
	parser.add_argument("--step_size", type=float, default=50, help="Step size for StepLR")
	parser.add_argument('--milestone', type=list, default=[100, 150], metavar='M', help='Milestone for MultiStepLR')
	parser.add_argument('--warmup', type=int, default=10, metavar='warmup', help='warmup epochs')
	parser.add_argument('--lr-mqnet', type=float, default=0.001, help='learning rate for updating mqnet')
	parser.add_argument('--epochs-mqnet', default=100, type=int, help='number of epochs for training mqnet')


	parser.add_argument('--mqnet-mode', default="CONF", help="specifiy the mode of MQNet to use") #CONF, LL
	parser.add_argument('--steps-per-epoch', type=int, default=100, metavar='N', help='number of steps per epoch')
	parser.add_argument('--epoch-loss', default=120, type=int, help='number of epochs for training loss module in LL')
	



	# parser.add_argument("--resize_factor", help='resize scale is sampled from [resize_factor, 1.0]',default=0.08, type=float)
	# parser.add_argument("--resize_fix", help='resize scale is fixed to resize_factor (not (resize_factor, 1.0])',action='store_true')

	# to do: consider add different AL methods

	#hyper parameters
	parser.add_argument('--train_epochs', type=int, default=100, help='Number of training epochs')
	
	args = parser.parse_args()
	
	return args


