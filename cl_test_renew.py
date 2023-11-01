import torch
import os
from torch.autograd import Variable
import torch.utils.data as torchdata
import torch.nn as nn
import numpy as np
import tqdm
import utils
import csv
import pandas as pd
import openpyxl
import matplotlib.pyplot as plt
import time

import torch.backends.cudnn as cudnn
cudnn.benchmark = True

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='R110_C10')
parser.add_argument('--data_dir', default='data/')
parser.add_argument('--load_dir', default=None)
parser.add_argument('--load', default=None)
parser.add_argument('--raw_dir', default=None)
args = parser.parse_args()

DIR = args.raw_dir
if not os.path.exists(args.raw_dir):
    os.system('mkdir -p ' + args.raw_dir + './img/')

if args.load_dir is not None:
    path = "./" + args.load_dir
    file_list = os.listdir(path)
    file_list_t7 = [path + file for file in file_list if file.endswith(".t7")]

else:
    path = "./" + args.load
    file_list_t7 = [path]




#---------------------------------------------------------------------------------------------#
class FConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(FConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)
        self.num_ops = 0

    def forward(self, x):
        output = super(FConv2d, self).forward(x)
        output_area = output.size(-1)*output.size(-2)
        filter_area = np.prod(self.kernel_size)
        self.num_ops += 2*self.in_channels*self.out_channels*filter_area*output_area
        return output

class FLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(FLinear, self).__init__(in_features, out_features, bias)
        self.num_ops = 0

    def forward(self, x):
        output = super(FLinear, self).forward(x)
        self.num_ops += 2*self.in_features*self.out_features
        return output

def count_flops(model, reset=True):
    op_count = 0
    for m in model.modules():
        if hasattr(m, 'num_ops'):
            op_count += m.num_ops
            if reset: # count and reset to 0
                m.num_ops = 0

    return op_count

# replace all nn.Conv and nn.Linear layers with layers that count flops
nn.Conv2d = FConv2d
nn.Linear = FLinear

#--------------------------------------------------------------------------------------------#
def log(accuracy, sparsity, variance, ops_mean, ops_std, leng, policy_set, currnet_model, timings):

    filename = DIR + "log_" + ''.join(map(str, currnet_model.split("/")[-1].split(".")[:-1]))
    timing_mean, timing_std = np.mean(timings), np.std(timings)
    agent_timing_mean, agent_timing_std = np.mean(timings[:,0]), np.std(timings[:,0])
    rnet_timing_mean, rnet_timing_std = np.mean(timings[:,1]), np.std(timings[:,1])

    df_time = pd.DataFrame(list([timing_mean, timing_std, agent_timing_mean, agent_timing_std, rnet_timing_mean, rnet_timing_std]))
    df_time_raw = pd.DataFrame(timings, columns=['agent_time', 'rnet_time'])
    
    log_df = pd.DataFrame({
        'Accuracy' : [accuracy.item()],
        'Block Usage' : [f'{sparsity} \u00B1 {variance}'],
        'FLOPs/img' : [f'{ops_mean} \u00B1 {ops_std}'],
        'Unique Policies' : [leng],
    })
    policy_df = pd.DataFrame({
        'policy_set' : list(policy_set)
    })
    with pd.ExcelWriter(filename + '.xlsx', engine='openpyxl') as writer:
        policy_df.to_excel(writer, args.model + '_log', startrow = 3, index=False)
        log_df.to_excel(writer, args.model + '_log', index=False)
        df_time.to_excel(writer, args.model + '_timing', index=False)
        df_time_raw.to_excel(writer, args.model + '_timing',startrow=3, index=False)
    #----save average image
    x = [idx for idx in range(54)]
    y = []
    result = [f"0" for _ in range(54)]
    for policy_idx, current_pol in enumerate(list(policy_set)):
        for idx, bit in enumerate(current_pol):
            result[idx] = str(int(result[idx]) + int(bit))
    y = [float(value)*100/len(policy_set) for value in result]
    print()
    if sum(y) == 0 :
        label_1 = 0
        label_2 = 0
        label_3 = 0
        label_4 = 0
    else:    
        label_1 = sum(y[:14]) * sparsity.item() / sum(y)
        label_2 = sum(y[14:27]) * sparsity.item() / sum(y)
        label_3 = sum(y[27:41]) * sparsity.item() / sum(y)
        label_4 = sum(y[41:]) * sparsity.item() / sum(y)
    plt.clf()
    plt.plot(x, y, marker='o', linestyle='-')  # 선 그래프 (마커 o, 실선)
    plt.xlabel('Block number')  # x 축 레이블
    plt.ylabel('Block Usage [%]')  # y 축 레이블
    plt.title(f'Block Usage Ratio Epoch:{currnet_model.split("/")[-1].split("_")[2]}')  # 그래프 제목
    plt.axvline(x=len(x)/4-1, color='red', linestyle=':')
    plt.axvline(x=len(x)/2, color='red', linestyle=':')
    plt.axvline(x=len(x)*3/4-1, color='red', linestyle=':')
    plt.text(len(x)/8-1, -1, round(label_1, 1), fontsize=10, color='red', ha='center', va='bottom')
    plt.text(len(x)*3/8-1, -1, round(label_2, 1), fontsize=10, color='red', ha='center', va='bottom')
    plt.text(len(x)*5/8-1, -1, round(label_3, 1), fontsize=10, color='red', ha='center', va='bottom')
    plt.text(len(x)*7/8-1, -1, round(label_4, 1), fontsize=10, color='red', ha='center', va='bottom')    
    plt.xlim(-1, 54)  # x 축 범위 설정 (최소값, 최대값)
    plt.ylim(-1, 100)  # y 축 범위 설정 (최소값, 최대값)
    plt.savefig(DIR + "img/" + ''.join(map(str, currnet_model.split("/")[-1].split(".")[:-1])) + ".png")

#--------------------------------------------------------------------------------------------------------#
def test():

    total_ops = []
    matches, policies, record_time = [], [], []
    for batch_idx, (inputs, targets) in tqdm.tqdm(enumerate(testloader), total=len(testloader.dataset)):
        
        with torch.no_grad():
            inputs = Variable(inputs).cuda()
        targets = Variable(targets).cuda()
        
        agent_start_event.record()
        probs, _ = agent(inputs)
        agent_end_event.record()
        torch.cuda.synchronize()
        agent_time = agent_start_event.elapsed_time(agent_end_event)

        policy_expend = torch.ones(probs.size(0), sum(rnet.layer_config) - agent.num_blocks).cuda()
        policy = torch.cat((probs.clone(),policy_expend), dim=1)
        policy[policy<0.5] = 0.0
        policy[policy>=0.5] = 1.0

        rnet_start_event.record()
        preds = rnet.forward_single(inputs, policy.data.squeeze(0))
        rnet_end_event.record()
        torch.cuda.synchronize()
        rnet_time = rnet_start_event.elapsed_time(rnet_end_event)
        
        _ , pred_idx = preds.max(1)
        match = (pred_idx==targets).data.float()

        matches.append(match)
        policies.append(policy.data)

        ops = count_flops(agent) + count_flops(rnet)
        total_ops.append(ops)
        record_time.append([agent_time, rnet_time])
    accuracy, _, sparsity, variance, policy_set = utils.performance_stats(policies, matches, matches)
    ops_mean, ops_std = np.mean(total_ops), np.std(total_ops)
    
    log_str = [
    f'Accuracy: {accuracy}',
    f'Block Usage: {sparsity} \u00B1 {variance}',
    f'FLOPs/img: {ops_mean} \u00B1 {ops_std}',
    f'Unique Policies: {len(policy_set)}'
    ]

    print(log_str)
    return accuracy, sparsity, variance, ops_mean, ops_std, len(policy_set), policy_set, record_time 

#--------------------------------------------------------------------------------------------------------#
trainset, testset = utils.get_dataset(args.model, args.data_dir)
testloader = torchdata.DataLoader(testset, batch_size=1, shuffle=False, num_workers=4)
rnet, agent = utils.get_model(args.model)
# if no model is loaded, use all blocks
agent.logit.weight.data.fill_(0)
agent.logit.bias.data.fill_(10)

print("loading checkpoints")

torch.cuda.empty_cache()

agent_start_event = torch.cuda.Event(enable_timing=True)
agent_end_event = torch.cuda.Event(enable_timing=True)
rnet_start_event = torch.cuda.Event(enable_timing=True)
rnet_end_event = torch.cuda.Event(enable_timing=True)

for currnet_model in tqdm.tqdm(file_list_t7, total=len(file_list_t7)):
    if args.load_dir is not None:
        utils.load_checkpoint(rnet, agent, currnet_model)

    rnet.eval().cuda()
    agent.eval().cuda()

    accuracy, sparsity, variance, ops_mean, ops_std, leng, policy_set, record_time = test()
    log(accuracy, sparsity, variance, ops_mean, ops_std, leng, policy_set, currnet_model, record_time)