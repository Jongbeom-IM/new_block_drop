import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import tqdm

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model', default='R110_C10')
parser.add_argument('--load_dir', default=None)
parser.add_argument('--load', default=None)
args = parser.parse_args()

def save_img():
    x = [idx for idx in range(54)]
    y = []
    y_avg = []
    result = [f"0" for _ in range(54)]

    for policy_idx, current_pol in enumerate(policy_set):
        for idx, bit in enumerate(current_pol):
            result[idx] = str(int(result[idx]) + int(bit))
        
        if (policy_idx + 1) % 1000 == 0:
            y = [float(item)/10 for item in result]
            #plt.clf()
            #plt.plot(x, y, marker='o', linestyle='-', label=f'Graph {int(policy_idx + 1) / 1000}')  # 선 그래프 (마커 o, 실선)
            #plt.xlabel('Block number')  # x 축 레이블
            #plt.ylabel('Block Usage [%]')  # y 축 레이블
            #plt.title(f'Block Usage Ratio out of 1000 Samples({policy_idx+1}/{len(policy_set)})')  # 그래프 제목
            #plt.xlim(-1, 54)  # x 축 범위 설정 (최소값, 최대값)
            #plt.ylim(-1, 100)  # y 축 범위 설정 (최소값, 최대값)
            ##plt.show()
            #plt.savefig(f'./img/{filename}_{int((policy_idx+1)/1000)}.png')  # 파일명 및 확장자 지정
            result = [f"0" for _ in range(54)]
            y_avg = y_avg + y
            if int(len(policy_set) / 1000) == int((policy_idx + 1) / 1000):
                y_avg = [float(value) / ((policy_idx + 1) / 1000) for value in y_avg ]
                plt.clf()
                plt.plot(x, y, marker='o', linestyle='-')  # 선 그래프 (마커 o, 실선)
                plt.xlabel('Block number')  # x 축 레이블
                plt.ylabel('Block Usage [%]')  # y 축 레이블
                plt.title(f'Block Usage Ratio out of 1000 Samples(average)')  # 그래프 제목
                plt.xlim(-1, 54)  # x 축 범위 설정 (최소값, 최대값)
                plt.ylim(-1, 100)  # y 축 범위 설정 (최소값, 최대값)
                plt.savefig("./img/a.png")
                #plt.savefig(f"./raw_data/R110_C10/img/{idx_xls}_average.png")

def save_img_avr():
    x = [idx for idx in range(54)]
    y = []
    result = [f"0" for _ in range(54)]
    for policy_idx, current_pol in enumerate(policy_set):
        for idx, bit in enumerate(current_pol):
            result[idx] = str(int(result[idx]) + int(bit))
    y = [float(value)*100/len(policy_set) for value in result]
    plt.clf()
    plt.plot(x, y, marker='o', linestyle='-')  # 선 그래프 (마커 o, 실선)
    plt.xlabel('Block number')  # x 축 레이블
    plt.ylabel('Block Usage [%]')  # y 축 레이블
    plt.title(f'Block Usage Ratio Epoch:{current_xlsx.split("/")[-1].split("_")[3]}')  # 그래프 제목
    plt.xlim(-1, 54)  # x 축 범위 설정 (최소값, 최대값)
    plt.ylim(-1, 100)  # y 축 범위 설정 (최소값, 최대값)
    plt.savefig(f"./raw_data/R110_C10/origin/img/{current_xlsx.split('/')[-1].split('.')[0]}.png")

if args.load_dir is not None:
    path = "./" + args.load_dir
    file_list = os.listdir(path)
    file_list_xlsx = [path + file for file in file_list if file.endswith(".xlsx")]
else:
    path = "./" + args.load
    file_list_xlsx = [path]

os.system('mkdir ./raw_data' + f"/{args.model}/img")

sheetname = 'R110_C10_log'
for idx_xls, current_xlsx in tqdm.tqdm(enumerate(file_list_xlsx), total=(len(file_list_xlsx))):
    data = pd.read_excel(current_xlsx,sheet_name=sheetname, skiprows=3)

    policy_set = [str(value) for value in data['policy_set']]

    for current_pol in policy_set:
        if len(current_pol) != len(policy_set[1]):
            print("data is not ok : "+current_xlsx)
        else :
            save_img_avr()

