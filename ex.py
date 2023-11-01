"""
import os

import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.utils.data as torchdata
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tqdm
import utils
import torch.optim as optim
from torch.distributions import Bernoulli
from tensorboard_logger import configure, log_value
import gc
from torch.utils.data import dataloader
import torch.backends.cudnn as cudnn
import csv
from torchsummary import summary
from models import resnet, base

model = "R110_C10"
data_dir = 'data/'
batch = 1
filename = "prob.csv"
rnet, agent = utils.get_model(model)


trainset, testset = utils.get_dataset(model, data_dir)
trainloader = dataloader.DataLoader(trainset, batch_size=batch, shuffle=True, num_workers=4)

input_size = (3, 224, 224)  # 입력 이미지 크기 (3 채널, 224x224 크기)
#summary(agent, input_size=input_size)
print("Model's state_dict:")
for param_tensor in agent.state_dict():
    print(param_tensor, "\t", agent.state_dict()[param_tensor].size())

print(sum(rnet.layer_config))

for index, (inputs, targets) in enumerate(trainloader):
    inputs, targets = Variable(inputs), Variable(targets).cuda(non_blocking=True)

    arr_input = np.array(inputs)
    probs, value = agent(inputs)
    policy_map = probs.data.clone()
    policy_map[policy_map<0.5] = 0.0
    policy_map[policy_map>=0.5] = 1.0
    policy_map = Variable(policy_map)
    policy_map_arr = np.array(policy_map.detach())
    probs_arr = np.array(probs.detach())
    value_arr = np.array(value.detach())
    print(len(arr_input),arr_input.shape)
    print(len(probs_arr),probs_arr.shape)
    print(len(policy_map_arr),policy_map_arr.shape)
    print(len(value_arr),value_arr.shape)
    with open(filename, mode='w', newline="") as log:
        writer = csv.writer(log)
        writer.writerows([["input"],inputs.data])
        writer.writerows([["probs"],probs])
        writer.writerows([["policy_map"],policy_map])
        writer.writerows([["value"],value])
 
    if index == 0:
        break
    """
import numpy 
import pandas as pd
import os
current_xlsx = '~/research/blockdrop/approach_model/log_ckpt_E_970_A_0742_R_383E-01_S_1693_#_5288.xlsx'

print(f"{''.join(map(str,current_xlsx.split('/')[-1].split('.')[:-1]))}")