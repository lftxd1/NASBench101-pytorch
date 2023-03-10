

import torch
from torch import nn


def Conv3_3(in_channels, out_channels,br=False):
    x=torch.nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1,  bias=True)
    if br:
        return x
    else:
        return torch.nn.Sequential(x,torch.nn.BatchNorm2d(out_channels),torch.nn.ReLU())

def Conv1_1(in_channels, out_channels,br=False):
    x=torch.nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0,  bias=True)
    if br:
        return x
    else:
        return torch.nn.Sequential(x,torch.nn.BatchNorm2d(out_channels),torch.nn.ReLU())
def MAXPOOL():
    return nn.MaxPool2d(kernel_size=(3, 3),stride=(1,1),padding=(1,1))
def DownSample(in_channels):
    return torch.nn.Sequential(nn.MaxPool2d(kernel_size=(2, 2),stride=(2,2),padding=(0,0)),Conv1_1(in_channels,int(in_channels*2)))


import torch
from torch import nn

import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader
import os
from PersonGenerate import *


#torch.cuda.set_device(0)

print(torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())

EPOCH = 100
BATCH_SIZE = 64
LR = 0.025

def cutout_transform(img, length: int = 16):
    h, w = img.size(1), img.size(2)
    mask = np.ones((h, w), np.float32)
    y = np.random.randint(h)
    x = np.random.randint(w)

    y1 = np.clip(y - length // 2, 0, h)
    y2 = np.clip(y + length // 2, 0, h)
    x1 = np.clip(x - length // 2, 0, w)
    x2 = np.clip(x + length // 2, 0, w)

    mask[y1: y2, x1: x2] = 0.
    mask = torch.from_numpy(mask)
    mask = mask.expand_as(img)
    img *= mask
    return img

CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

transform_with_cutout = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    cutout_transform,
])
transform_valid = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
])
INPUT = 'input'
OUTPUT = 'output'
CONV3X3 = 'conv3x3-bn-relu'
CONV1X1 = 'conv1x1-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'
CONV5X5= 'conv3x3-bn-relu'


# matrix=[  [0, 1, 0, 0, 0, 0, 0],    # input layer
#           [0, 0, 1, 0, 0, 0, 0],    # 1x1 conv
#           [0, 0, 0, 1, 0, 0, 0],    # 1x1 conv
#           [0, 0, 0, 0, 1, 0, 0],    # 3x3 conv (replaced by two 3x3's)
#           [0, 0, 0, 0, 0, 1, 0],    # 3x3 conv (replaced by two 3x3's)
#           [0, 0, 0, 0, 0, 0, 1],    # max-pool
#           [0, 0, 0, 0, 0, 0, 0]]    # output layer
# ops=[INPUT, CONV1X1, CONV3X3, CONV1X1+"1", CONV3X3+"1", MAXPOOL3X3, OUTPUT]

# matrix=[[0, 1, 1, 1, 0, 1, 0],    # input layer
#         [0, 0, 0, 0, 0, 0, 1],    # 1x1 conv
#         [0, 0, 0, 0, 0, 0, 1],    # 3x3 conv
#         [0, 0, 0, 0, 1, 0, 0],    # 1x1 conv (replaced by two 3x3's)
#         [0, 0, 0, 0, 0, 0, 1],    # 3x3 conv (replaced by two 3x3's)
#         [0, 0, 0, 0, 0, 0, 1],    # max-pool
#         [0, 0, 0, 0, 0, 0, 0]]   # output layer

# ops=[INPUT, CONV1X1, CONV3X3, CONV1X1+"1", CONV3X3+"1", MAXPOOL3X3, OUTPUT]
# i,j的定义是i->j
# 每个节点都有多/单个输入和一个输出

(matrix,ops)=GeneratePerson()

class Cell(nn.Module):
    def __init__(self,in_channels=128) -> None:
        super(Cell,self).__init__()


        self.reduce_layer=torch.nn.ModuleDict()
        self.normal_layer=torch.nn.ModuleDict()
        self.one_layer=torch.nn.ModuleDict()
        self.reduce_layer['maxpool3x3']=torch.nn.Sequential(MAXPOOL(),Conv1_1(in_channels,int(in_channels/2),True))
        self.normal_layer['maxpool3x3']=MAXPOOL()
        
        for i in range(5):
            name=""
            if i!=0:
                name="#"+str(i)
            self.reduce_layer["conv1x1-bn-relu"+name]=Conv1_1(in_channels,int(in_channels/2),True)
            self.reduce_layer["conv3x3-bn-relu"+name]=Conv3_3(in_channels,int(in_channels/2),True)
            self.reduce_layer['maxpool3x3'+name]=self.reduce_layer['maxpool3x3']
            self.normal_layer["conv1x1-bn-relu"+name]=Conv1_1(int(in_channels/2),int(in_channels/2),True)
            self.normal_layer["conv3x3-bn-relu"+name]=Conv3_3(int(in_channels/2),int(in_channels/2),True)
            self.normal_layer['maxpool3x3'+name]=MAXPOOL()

        self.proj=Conv1_1(in_channels,in_channels,False)

        for i in range(1,6):
            self.one_layer[str(int(in_channels/2)*i)]=Conv1_1(int((in_channels/2)*i),in_channels)

    def op(self,key,index):
        if index==0:
            return self.reduce_layer[key]
        else:
            return self.normal_layer[key]

    def forward(self,x,matrix,ops):
        output=[ 0 for i in range(len(matrix))]
        output[0]=x
        flag=[False]* len(matrix) # 是否该节点已经输出
        flag[0]=True
        pre_list=[ [] for i in range(len(matrix))]

        for node_index in range(len(matrix)):

            if node_index==len(matrix)-1:# 如果到达最后节点
                combine_list=[]
                #print(pre_list[node_index])
                for index in pre_list[node_index]:
                    if index==0:
                        output[node_index]=self.proj(output[index])
                        flag[node_index]=True
                    else:
                        #print(output[index].shape)
                        if flag[index]:
                            combine_list.append(output[index])
                if len(combine_list)!=0:
                    cat_res=torch.cat(combine_list,dim=1)

                    #print("___-")
                    #print(cat_res.shape)
                    if cat_res.shape!=x.shape:
                        cat_res=self.one_layer[str(cat_res.shape[1])](cat_res)
                    if flag[node_index]:
                        output[node_index]=output[node_index]+cat_res
                    else:
                        output[node_index]=cat_res
                return output[node_index]
            elif node_index!=0:
                for index in pre_list[node_index]:# index 是这个节点前置节点的索引
                    #print(output[index].shape)
                    if flag[index]==True:
                        op_name=ops[node_index]
                        #print(self.op(op_name,index))
                        output[node_index]=output[node_index]+self.op(op_name,index)(output[index])
                        flag[node_index]=True
                    else:
                        pass
                        #print(index,"->",node_index,"无前置节点")
                    
            for j in range(node_index+1,len(matrix[node_index])):
                if matrix[node_index][j]!=0:
                    pre_list[j].append(node_index)
        
def Conv3_3(in_channels, out_channels,br=False):
    x=torch.nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1,  bias=True)
    if br:
        return x
    else:
        return torch.nn.Sequential(x,torch.nn.BatchNorm2d(out_channels),torch.nn.ReLU())

def Conv1_1(in_channels, out_channels,br=False):
    x=torch.nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0,  bias=True)
    if br:
        return x
    else:
        return torch.nn.Sequential(x,torch.nn.BatchNorm2d(out_channels),torch.nn.ReLU())
def MAXPOOL():
    return nn.MaxPool2d(kernel_size=(3, 3),stride=(1,1),padding=(1,1))
def DownSample(in_channels):
    return torch.nn.Sequential(nn.MaxPool2d(kernel_size=(2, 2),stride=(2,2),padding=(0,0)),Conv1_1(in_channels,int(in_channels*2)))

class NasBench101Net(nn.Module):
    def __init__(self) -> None:
        super(NasBench101Net,self).__init__()
        self.stem=torch.nn.Sequential(Conv3_3(3,128,True))
        self.cell1=Cell(128)
        self.cell2=Cell(128)
        self.cell3=Cell(128)
        self.downsample1=DownSample(128)
        self.cell4=Cell(256)
        self.cell5=Cell(256)
        self.cell6=Cell(256)
        self.downsample2=DownSample(256)
        self.cell7=Cell(512)
        self.cell8=Cell(512)
        self.cell9=Cell(512)
        self.gl=torch.nn.AdaptiveAvgPool2d(1)
        self.flatten=nn.Flatten()
        self.dense=nn.Linear(512,10)
    def forward(self,x,matrix,op):
        x=self.stem(x)
        #print(x.shape)
        x=self.cell1(x,matrix,op)
        #print(x.shape)
        x=self.cell2(x,matrix,op)
        x=self.cell3(x,matrix,op)
        x=self.downsample1(x)
        x=self.cell4(x,matrix,op)
        x=self.cell5(x,matrix,op)
        x=self.cell6(x,matrix,op)
        x=self.downsample2(x)
        x=self.cell7(x,matrix,op)
        x=self.cell8(x,matrix,op)
        x=self.cell9(x,matrix,op)
        x=self.gl(x)
        x=self.flatten(x)
        x=self.dense(x)
        return x