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

import torch.backends.cudnn as cudnn
cudnn.benchmark = True

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model', default='R110_C10')
parser.add_argument('--data_dir', default='data/')
parser.add_argument('--load', default=None)
args = parser.parse_args()

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
def log(accuracy, sparsity, variance, ops_mean, ops_std, leng, policy_set):
    filename = "log_" + args.load.split("/")[-1]
    log_df = pd.DataFrame({
        'Accuracy' : [accuracy.item()],
        'Block Usage' : [f'{sparsity} \u00B1 {variance}'],
        'FLOPs/img' : [f'{ops_mean} \u00B1 {ops_std}'],
        'Unique Policies' : [leng],
    })
    policy_df = pd.DataFrame({
        'policy_set' : list(policy_set)
    })
    with pd.ExcelWriter(filename + ".xlsx", engine='openpyxl') as writer:
        policy_df.to_excel(writer, args.model + "_log", startrow = 3, index=False)
        log_df.to_excel(writer, args.model + "_log", index=False)
#--------------------------------------------------------------------------------------------------------#
def test():

    total_ops = []
    matches, policies = [], []
    for batch_idx, (inputs, targets) in tqdm.tqdm(enumerate(testloader), total=len(testloader.dataset)):
        
        with torch.no_grad():
            inputs = Variable(inputs).cuda()
        targets = Variable(targets).cuda()
        probs, _ = agent(inputs)

        policy = probs.clone()
        policy[policy<0.5] = 0.0
        policy[policy>=0.5] = 1.0

        start_event.record()
        preds = rnet.forward_single(inputs, policy.data.squeeze(0))
        end_event.record()
        
        _ , pred_idx = preds.max(1)
        match = (pred_idx==targets).data.float()
        elapsed_time_ms = start_event.elapsed_time(end_event)
        print(elapsed_time_ms)

        matches.append(match)
        policies.append(policy.data)

        ops = count_flops(agent) + count_flops(rnet)
        total_ops.append(ops)

    accuracy, _, sparsity, variance, policy_set = utils.performance_stats(policies, matches, matches)
    ops_mean, ops_std = np.mean(total_ops), np.std(total_ops)
#    policy_df = pd.DataFrame({
#                'policy_set' : list(policy_set)
#                })
#    reward_df['accuracy'].to_excel(log, sheet_name='20231010_R110C10', index=False, startcol=0)
#    reward_df['sparsity'].to_excel(log, sheet_name='20231010_R110C10', index=False, startcol=2)
#    reward_df['variance'].to_excel(log, sheet_name='20231010_R110C10', index=False, startcol=3)
#    reward_df['policy_set'].to_excel(log, sheet_name='20231010_R110C10_policy', index=False)
    
    log_str = [
    f'Accuracy: {accuracy}',
    f'Block Usage: {sparsity} \u00B1 {variance}',
    f'FLOPs/img: {ops_mean} \u00B1 {ops_std}',
    f'Unique Policies: {len(policy_set)}'
    ]

    print(log_str)
    return accuracy, sparsity, variance, ops_mean, ops_std, len(policy_set), policy_set  

#--------------------------------------------------------------------------------------------------------#
trainset, testset = utils.get_dataset(args.model, args.data_dir)
testloader = torchdata.DataLoader(testset, batch_size=1, shuffle=False, num_workers=4)
rnet, agent = utils.get_model(args.model)
# if no model is loaded, use all blocks
agent.logit.weight.data.fill_(0)
agent.logit.bias.data.fill_(10)

print("loading checkpoints")

torch.cuda.empty_cache()

if args.load is not None:
    utils.load_checkpoint(rnet, agent, args.load)

rnet.eval().cuda()
agent.eval().cuda()


start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

accuracy, sparsity, variance, ops_mean, ops_std, leng, policy_set = test()
log(accuracy, sparsity, variance, ops_mean, ops_std, leng, policy_set)