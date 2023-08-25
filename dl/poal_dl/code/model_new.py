import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import os
import numpy as np
import torch
from torch import set_grad_enabled
from abc import *


def get_net(name, net_name, task_name):
	if net_name == 'ResNet18':
		if 'CIFAR' in name:
			return ResNet18_cifar100
		else:
			raise NotImplementedError
	else:
		raise NotImplementedError

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class ResNet18_cifar100(nn.Module):
	def __init__(self, dim = 28 * 28, pretrained=False, num_classes = 10):
		super().__init__()
		resnet18 = models.resnet18(pretrained=pretrained)
		features_tmp = nn.Sequential(*list(resnet18.children())[:-1]) # without the last year
		#print(features_tmp)
		features_tmp[0] = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
		#print(features_tmp)
		self.features = nn.Sequential(*list(features_tmp))
		#self.features = nn.Sequential(*list(features_tmp)[0:3], *list(features_tmp)[4:-1])
		self.feature0 = nn.Sequential(*list(features_tmp)[0:4])
		self.feature1 = nn.Sequential(*list(features_tmp)[4])
		self.feature2 = nn.Sequential(*list(features_tmp)[5])
		self.feature3 = nn.Sequential(*list(features_tmp)[6]) 
		self.feature4 = nn.Sequential(*list(features_tmp)[7])
		self.feature5 = nn.Sequential(*list(features_tmp)[8:9])

		self.classifier = nn.Linear(512, num_classes)
		self.dim = resnet18.fc.in_features
		
	
	def forward(self, x):
		feature  = self.features(x)
		#print('feature', feature.shape)
		x = feature.view(feature.size(0), -1)		
		#print(x.shape)
		output = self.classifier(x)
		return output, x
	
	def feature_list(self, x):
		out_list = []
		out = self.feature0(x)
		out_list.append(out)
		out = self.feature1(out)
		out_list.append(out)
		out = self.feature2(out)
		out_list.append(out)
		out = self.feature3(out)
		out_list.append(out)
		out = self.feature4(out)
		out_list.append(out)
		out = self.feature5(out)
		out = out.view(out.size(0), -1)		
		y = self.classifier(out)
		return y, out_list

	def intermediate_forward(self, x, layer_index):
		out = self.feature0(x)
		if layer_index == 1:
			out = self.feature1(out)
		elif layer_index == 2:
			out = self.feature1(out)
			out = self.feature2(out)
		elif layer_index == 3:
			out = self.feature1(out)
			out = self.feature2(out)
			out = self.feature3(out)
		elif layer_index == 4:
			out = self.feature1(out)
			out = self.feature2(out)
			out = self.feature3(out)
			out = self.feature4(out)
		return out

	def penultimate_forward(self, x):
		out = self.feature0(x)
		out = self.feature1(out)
		out = self.feature2(out)
		out = self.feature3(out)
		penultimate = self.feature4(out)
		out = self.feature5(penultimate)
		out = out.view(out.size(0), -1)		
		y = self.classifier(out)
		return y, penultimate

	def get_embedding_dim(self):
		return self.dim


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet18_LL(nn.Module):
    def __init__(self, channel=3, num_classes=10, record_embedding: bool = False,
                 no_grad: bool = False):
        super().__init__()
        self.in_planes = 64
        block = BasicBlock
        num_blocks = [2, 2, 2, 2]

        self.conv1 = conv3x3(channel, 64)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes, bias=False)

        self.embedding_recorder = EmbeddingRecorder(record_embedding)
        self.no_grad = no_grad

    def get_last_layer(self):
        return self.linear

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        with set_grad_enabled(not self.no_grad):
            out = F.relu(self.bn1(self.conv1(x)))
            out1 = self.layer1(out)
            out2 = self.layer2(out1)
            out3 = self.layer3(out2)
            out4 = self.layer4(out3)
            out = F.avg_pool2d(out4, 4)
            out_cnn = out.view(out.size(0), -1)
            out = self.embedding_recorder(out_cnn)
            out = self.linear(out)
        return out, [out1, out2, out3, out4]
    
class EmbeddingRecorder(nn.Module):
    def __init__(self, record_embedding: bool = False):
        super().__init__()
        self.record_embedding = record_embedding

    def forward(self, x):
        if self.record_embedding:
            self.embedding = x
        return x

    def __enter__(self):
        self.record_embedding = True

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.record_embedding = False

class LossNet32(nn.Module):
    def __init__(self, feature_sizes=[32, 16, 8, 4], num_channels=[64, 128, 256, 512], interm_dim=128):
        super().__init__()

        self.GAP1 = nn.AvgPool2d(feature_sizes[0])
        self.GAP2 = nn.AvgPool2d(feature_sizes[1])
        self.GAP3 = nn.AvgPool2d(feature_sizes[2])
        self.GAP4 = nn.AvgPool2d(feature_sizes[3])

        self.FC1 = nn.Linear(num_channels[0], interm_dim)
        self.FC2 = nn.Linear(num_channels[1], interm_dim)
        self.FC3 = nn.Linear(num_channels[2], interm_dim)
        self.FC4 = nn.Linear(num_channels[3], interm_dim)

        self.linear = nn.Linear(4 * interm_dim, 1)

    def forward(self, features):
        out1 = self.GAP1(features[0])
        out1 = out1.view(out1.size(0), -1)
        out1 = F.relu(self.FC1(out1))

        out2 = self.GAP2(features[1])
        out2 = out2.view(out2.size(0), -1)
        out2 = F.relu(self.FC2(out2))

        out3 = self.GAP3(features[2])
        out3 = out3.view(out3.size(0), -1)
        out3 = F.relu(self.FC3(out3))

        out4 = self.GAP4(features[3])
        out4 = out4.view(out4.size(0), -1)
        out4 = F.relu(self.FC4(out4))

        out = self.linear(torch.cat((out1, out2, out3, out4), 1))
        return out
    


class BaseModel(nn.Module, metaclass=ABCMeta):
    def __init__(self, last_dim, num_classes=10, simclr_dim=128):
        super(BaseModel, self).__init__()
        self.linear = nn.Linear(last_dim, num_classes)
        self.simclr_layer = nn.Sequential(
            nn.Linear(last_dim, last_dim),
            nn.ReLU(),
            nn.Linear(last_dim, simclr_dim),
        )
        self.shift_cls_layer = nn.Linear(last_dim, 4)
        self.joint_distribution_layer = nn.Linear(last_dim, 4 * num_classes)
        '''
        self.shift_dis = nn.Sequential(
            nn.Linear(simclr_dim*2, simclr_dim),
            nn.ReLU(),
            nn.Linear(simclr_dim, 1),
            nn.Sigmoid()
        )
        '''

    @abstractmethod
    def penultimate(self, inputs, all_features=False):
        pass

    def forward(self, inputs=None, penultimate=False, simclr=False, shift=False, joint=False):
        _aux = {}
        _return_aux = False

        features = self.penultimate(inputs)
        output = self.linear(features)

        if penultimate:
            _return_aux = True
            _aux['penultimate'] = features

        if simclr:
            _return_aux = True
            _aux['simclr'] = self.simclr_layer(features)

        if shift:
            _return_aux = True
            _aux['shift'] = self.shift_cls_layer(features)

        if joint:
            _return_aux = True
            _aux['joint'] = self.joint_distribution_layer(features)

        if _return_aux:
            return output, _aux

        return output
    

class ResNet_CSI(BaseModel):
    def __init__(self, channel=3, num_classes=10, record_embedding: bool = False,
                 no_grad: bool = False):
        block = BasicBlock
        last_dim = 512 * block.expansion
        num_blocks = [2, 2, 2, 2]
        super(ResNet_CSI, self).__init__(last_dim, num_classes)

        self.in_planes = 64
        self.last_dim = last_dim

        self.normalize = NormalizeLayer()

        self.conv1 = conv3x3(3, 64)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def penultimate(self, x, all_features=False):
        out_list = []

        out = self.normalize(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = F.relu(out)
        out_list.append(out)

        out = self.layer1(out)
        out_list.append(out)
        out = self.layer2(out)
        out_list.append(out)
        out = self.layer3(out)
        out_list.append(out)
        out = self.layer4(out)
        out_list.append(out)

        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)

        if all_features:
            return out, out_list
        else:
            return out
        

class NormalizeLayer(nn.Module):
    """
    In order to certify radii in original coordinates rather than standardized coordinates, we
    add the Gaussian noise _before_ standardizing, which is why we have standardization be the first
    layer of the classifier rather than as a part of preprocessing as is typical.
    """

    def __init__(self):
        super(NormalizeLayer, self).__init__()

    def forward(self, inputs):
        return (inputs - 0.5) / 0.5
    

class QueryNet(nn.Module):
    def __init__(self, input_size=2, inter_dim=64):
        super().__init__()

        W1 = torch.rand(input_size, inter_dim, requires_grad=True) #ones
        W2 = torch.rand(inter_dim, 1, requires_grad=True) #ones
        b1 = torch.rand(inter_dim, requires_grad=True) #zeros
        b2 = torch.rand(1, requires_grad=True) #zeros

        self.W1 = torch.nn.parameter.Parameter(W1, requires_grad=True)
        self.W2 = torch.nn.parameter.Parameter(W2, requires_grad=True)
        self.b1 = torch.nn.parameter.Parameter(b1, requires_grad=True)
        self.b2 = torch.nn.parameter.Parameter(b2, requires_grad=True)

        #print(self.W2) # all 1

    def forward(self, X):
        out = torch.sigmoid(torch.matmul(X, torch.relu(self.W1)) + self.b1)
        out = torch.matmul(out, torch.relu(self.W2)) + self.b2
        return out

    


