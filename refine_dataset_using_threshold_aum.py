import os
import argparse
import csv
import ast
from tqdm import tqdm
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str)
parser.add_argument('--data_type', type=str)
parser.add_argument('--model_type', type=str)
parser.add_argument('--data_path', type=str)
parser.add_argument('--sampling_ratio', type=float)
args = parser.parse_args()

print(args)



original_path = './output/'+args.task+'_'+args.model_type+'_original_'+args.data_type+'/aum_values.csv'
threshold_path = './output/'+args.task+'_'+args.model_type+'_threshold_'+args.data_type+'/aum_values.csv'
data_path = args.data_path


class SNLIProcessor:
    def load_samples(self, path):
        samples = {}
        with open(path, newline='') as f:
            reader = csv.reader(f, delimiter='\t')
            desc = f'loading \'{path}\''
            idx = 0
            for row in tqdm(reader, desc=desc):
                if idx == 0: 
                    header = row
                else:
                    guid = row[1]
                    samples[guid] = row
                idx += 1
        return samples, header

class QQPProcessor:
    def load_samples(self, path):
        samples = {}
        with open(path, newline='') as f:
            reader = csv.reader(f, delimiter='\t')
            desc = f'loading \'{path}\''
            idx = 0
            for row in tqdm(reader, desc=desc):
            #for row in reader:
                if idx == 0:
                    header = row
                else:
                    guid = row[0]
                    samples[guid] = row
                idx += 1
        return samples, header

class SWAGProcessor:
    def load_samples(self, path):
        samples = {}
        with open(path, newline='') as f:
            reader = csv.reader(f, delimiter='\t')
            desc = f'loading \'{path}\''
            idx = 0
            for row in tqdm(reader, desc=desc):
                if idx == 0: 
                    header = row
                else:
                    guid = row[5]
                    samples[guid] = row
                idx += 1
        return samples, header

def select_processor():
    """Selects data processor using task name."""
    return globals()[f'{args.task}Processor']()

processor = select_processor()
data, header = processor.load_samples(data_path)

#aums = []
th_aum_dict = {}
or_aum_dict = {}
with open(threshold_path) as f:
    reader = csv.reader(f, delimiter=',')
    desc = f'loading \'{threshold_path}\''
    th_aums = []
    idx = 0
    for row in reader:
        if idx == 0: 
            idx += 1
            continue
        guid = int(row[0])
        aum = float(row[1])
        val_dict = {}
        val_dict['aum'] = aum
        th_aum_dict[guid] = val_dict
        idx += 1
        th_aums.append(aum)
        #aums.append(-aum)
with open(original_path) as f:
    reader = csv.reader(f, delimiter=',')
    desc = f'loading \'{original_path}\''
    or_aums = []
    idx = 0
    for row in reader:
        if idx == 0: 
            idx += 1
            continue
        guid = int(row[0])
        aum = float(row[1])
        val_dict = {}
        val_dict['aum'] = aum
        or_aum_dict[guid] = val_dict
        idx += 1
        or_aums.append(aum)
        #aums.append(aum)

sns.set_style('whitegrid')
plot = sns.kdeplot(np.array(th_aums), bw=0.5, label='Threhold Examples')
plot = sns.kdeplot(np.array(or_aums), bw=0.5, label='Original Examples')
plot.legend()
plt.savefig(args.task+"_"+args.model_type+"_"+args.data_type+'_output.png')

length = len(th_aums)
th_aums.sort()
th_aums = th_aums[:int(length*args.sampling_ratio)]
alpha = max(th_aums)

refine_data = []
filter_data = []
for key in data:
    row = data[key]
    if int(key) in or_aum_dict:
        aum = or_aum_dict[int(key)]['aum']
        if aum > alpha:
            refine_data.append(row)
        else:
            filter_data.append(row)


if '.tsv' in args.data_path:
    new_data_path = args.data_path.replace(".tsv","")+"_woMislabeled"+".tsv"
elif '.txt' in args.data_path:
    new_data_path = args.data_path.replace(".txt","")+"_woMislabeled"+".txt"


with open(new_data_path, 'w', newline='',encoding='utf-8') as output_file:
    output_file.write(str(header)+'\n')
    desc = f'writing \'{new_data_path}\''
    for item in tqdm(refine_data,desc=desc):
        output_file.write(str("\t".join(item))+'\n')



print("The original data # of instances")
print(len(data))
print("The refined data with using threshold instances")
print(len(refine_data))


