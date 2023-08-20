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
	parser.add_argument('--dataset', '-d', default='CIFAR10', type=str, help='dataset name')
	parser.add_argument('--iteration', '-t', default=3, type=int, help='time of repeat the experiment')
	parser.add_argument('--data_path', type=str, default='./../data', help='Path to where the data is')
	parser.add_argument('--out_path', type=str, default='./../results', help='Path to where the output log will be')
	parser.add_argument('--log_name', type=str, default='test.log', help='middle outputs')
	#parser.add_argument('--help', '-h', default=False, action='store_true', help='verbose')
	parser.add_argument('--cuda', action='store_true', help='If training is to be done on a GPU')
	parser.add_argument('--model', '-m', default='ResNet18', type=str, help='dataset name')
	parser.add_argument('--initseed', '-s', default = 1000, type = int, help = 'Initial pool of labeled data') # the number of data in the labled pool
	parser.add_argument('--gpu', '-g', default = 4, type = int, help = 'which gpu') # the number of GPU
	parser.add_argument('--seed', default=4666, type=int, help='random seed')
	parser.add_argument('--num_class', default=10, type=int, help='number of image classes')

	# parameters for mqnet
	parser.add_argument("--resize_factor", help='resize scale is sampled from [resize_factor, 1.0]',default=0.08, type=float)
	parser.add_argument("--resize_fix", help='resize scale is fixed to resize_factor (not (resize_factor, 1.0])',action='store_true')

	# to do: consider add different AL methods

	#hyper parameters
	parser.add_argument('--train_epochs', type=int, default=100, help='Number of training epochs')
	
	args = parser.parse_args()
	
	return args


