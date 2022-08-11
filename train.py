# Taken from https://github.com/shreydesai/calibration.
import argparse
import csv
import os
import sys
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
from transformers import AdamW, AutoModel, AutoTokenizer,get_linear_schedule_with_warmup
from tqdm import tqdm
from aum import *
import pickle
from itertools import cycle
import random

csv.field_size_limit(sys.maxsize)


parser = argparse.ArgumentParser()
parser.add_argument('--device', type=int, default=0, help='CUDA device')
parser.add_argument('--model', type=str, help='pre-trained model (bert-base-uncased, roberta-base)')
parser.add_argument('--task', type=str, help='task name (SNLI, MNLI, QQP, TwitterPPDB, SWAG, HellaSWAG)')
parser.add_argument('--max_seq_length', type=int, default=256, help='max sequence length')
parser.add_argument('--ckpt_path', type=str, help='model checkpoint path')
parser.add_argument('--output_path', type=str, help='model output path')
parser.add_argument('--train_path', type=str, help='train dataset path')
parser.add_argument('--dev_path', type=str, help='dev dataset path')
parser.add_argument('--test_path', type=str, help='test dataset path')
parser.add_argument('--epochs', type=int, default=3, help='number of epochs')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--learning_rate', type=float, default=1.0708609960508476e-05, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0., help='weight decay')
parser.add_argument('--max_grad_norm', type=float, default=1., help='gradient clip')
parser.add_argument('--do_train', action='store_true', default=False, help='enable training')
parser.add_argument('--do_evaluate', action='store_true', default=False, help='enable evaluation')
parser.add_argument('--mixup',action='store_true', default=False, help='enable mixup')
parser.add_argument('--mixup_type', type=str, help='select data type for mixup')
parser.add_argument('--hard_train_path', type=str, help='hard to learn train dataset path')
parser.add_argument('--easy_train_path', type=str, help='easy to learn train dataset path')
parser.add_argument('--ambig_train_path', type=str, help='ambiguous train dataset path')
parser.add_argument('--warmup_steps',type=int, default=0)
parser.add_argument('--gradient_accumulation_steps',default=1)
parser.add_argument('--ls',action='store_true',help='enable label smoothing')
parser.add_argument('--grad_extract', action='store_true', help='magnitude of gradients measuring method')
parser.add_argument('--aum', action='store_true')
parser.add_argument('--threshold_sample',action='store_true')
args = parser.parse_args()
print(args)


assert args.task in ('SNLI', 'MNLI', 'QQP', 'TwitterPPDB', 'SWAG', 'HellaSWAG')
assert args.model in ('bert-base-uncased', 'roberta-base')


if args.task in ('SNLI', 'MNLI'):
    n_classes = 3
elif args.task in ('QQP', 'TwitterPPDB'):
    n_classes = 2
elif args.task in ('SWAG', 'HellaSWAG'):
    n_classes = 1

if args.threshold_sample:
    if args.task in ('SNLI', 'MNLI', 'QQP', 'TwitterPPDB'):
        n_classes += 1


if args.threshold_sample:
    save_dir_th = './output/'+args.task +'_' +args.model + "_threshold"
    save_dir_original = './output/'+args.task +'_' +args.model + "_original"
    if 'ambig' in args.train_path: 
        save_dir_th = save_dir_th + '_ambig'
        save_dir_original = save_dir_original + '_ambig'
    elif 'easy' in args.train_path:
        save_dir_th = save_dir_th + '_easy'
        save_dir_original = save_dir_original + '_easy'
    elif 'hard' in args.train_path:
        save_dir_th = save_dir_th + '_hard'
        save_dir_original = save_dir_original + '_hard'

    if not os.path.exists(save_dir_th):
        os.mkdir(save_dir_th)
    if not os.path.exists(save_dir_original):
        os.mkdir(save_dir_original)
    aum_calculator_original = AUMCalculator(save_dir_original,compressed=False)
    aum_calculator_th = AUMCalculator(save_dir_th,compressed=False)


if args.aum:
    save_dir = './output/'+args.task +'_' +args.model
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    aum_calculator = AUMCalculator(save_dir,compressed=False)

def cuda(tensor):
    """Places tensor on CUDA device."""
    if args.device==-1:
        return tensor
    else:
        return tensor.to(args.device)


def load(dataset, batch_size, shuffle):
    """Creates data loader with dataset and iterator options."""

    return DataLoader(dataset, batch_size, shuffle=shuffle)


def adamw_params(model):
    """Prepares pre-trained model parameters for AdamW optimizer."""

    no_decay = ['bias', 'LayerNorm.weight']
    params = [
        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay
        },
        {
            'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        },
    ]
    return params


def encode_pair_inputs(sentence1, sentence2):
    """
    Encodes pair inputs for pre-trained models using the template
    [CLS] sentence1 [SEP] sentence2 [SEP]. Used for SNLI, MNLI, QQP, and TwitterPPDB.
    Returns input_ids, segment_ids, and attention_mask.
    """

    inputs = tokenizer.encode_plus(
        sentence1, sentence2, add_special_tokens=True, max_length=args.max_seq_length
    )
    input_ids = inputs['input_ids']
    if args.model == 'bert-base-uncased':
        segment_ids = inputs['token_type_ids']
    else:
        segment_ids = [0]*len(inputs['input_ids'])
    attention_mask = inputs['attention_mask']
    padding_length = args.max_seq_length - len(input_ids)
    input_ids += [tokenizer.pad_token_id] * padding_length
    segment_ids += [0] * padding_length
    attention_mask += [0] * padding_length
    for input_elem in (input_ids, segment_ids, attention_mask):
        assert len(input_elem) == args.max_seq_length
    return (
        cuda(torch.tensor(input_ids)).long(),
        cuda(torch.tensor(segment_ids)).long(),
        cuda(torch.tensor(attention_mask)).long(),
    )


def encode_mc_inputs(context, start_ending, endings):
    """
    Encodes multiple choice inputs for pre-trained models using the template
    [CLS] context [SEP] ending_i [SEP] where 0 <= i < len(endings). Used for
    SWAG and HellaSWAG. Returns input_ids, segment_ids, and attention_masks.
    """

    all_input_ids = []
    all_segment_ids = []
    all_attention_masks = []
    for ending in endings:
        inputs = tokenizer.encode_plus(
            context, start_ending+" " + ending, add_special_tokens=True, max_length=args.max_seq_length
        )
        input_ids = inputs['input_ids']
        if args.model == 'bert-base-uncased':
            segment_ids = inputs['token_type_ids']
        else:
            segment_ids = [0] * len(inputs['input_ids'])
        attention_mask = inputs['attention_mask']
        padding_length = args.max_seq_length - len(input_ids)
        input_ids += [tokenizer.pad_token_id] * padding_length
        segment_ids += [0] * padding_length
        attention_mask += [0] * padding_length
        for input_elem in (input_ids, segment_ids, attention_mask):
            assert len(input_elem) == args.max_seq_length
        all_input_ids.append(input_ids)
        all_segment_ids.append(segment_ids)
        all_attention_masks.append(attention_mask)
    return (
        cuda(torch.tensor(all_input_ids)).long(),
        cuda(torch.tensor(all_segment_ids)).long(),
        cuda(torch.tensor(all_attention_masks)).long(),
    )


def encode_label(label):
    """Wraps label in tensor."""

    return cuda(torch.tensor(label)).long()


class SNLIProcessor:
    """Data loader for SNLI."""

    def __init__(self):
        self.label_map = {'entailment': 0, 'contradiction': 1, 'neutral': 2}

    def valid_inputs(self, sentence1, sentence2, label):
        return len(sentence1) > 0 and len(sentence2) > 0 and label in self.label_map

    def load_samples(self, path):
        samples = []
        with open(path, newline='') as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader)  # skip header
            desc = f'loading \'{path}\''
            for row in tqdm(reader, desc=desc):
                try:
                    guid = row[1]
                    sentence1 = row[4]
                    sentence2 = row[7]
                    label = row[2]
                    if self.valid_inputs(sentence1, sentence2, label):
                        label = self.label_map[label]
                        samples.append((sentence1, sentence2, label, guid))
                except:
                    pass
        return samples


class MNLIProcessor(SNLIProcessor):
    """Data loader for MNLI."""

    def load_samples(self, path):
        samples = []
        with open(path, newline='') as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader)  # skip header
            desc = f'loading \'{path}\''
            for row in tqdm(reader, desc=desc):
                try:
                    sentence1 = row[8]
                    sentence2 = row[9]
                    label = row[-1]
                    if self.valid_inputs(sentence1, sentence2, label):
                        label = self.label_map[label]
                        samples.append((sentence1, sentence2, label, []))
                except:
                    pass
        return samples


class QQPProcessor:
    """Data loader for QQP."""

    def valid_inputs(self, sentence1, sentence2, label):
        return len(sentence1) > 0 and len(sentence2) > 0 and label in ('0', '1')

    def load_samples(self, path):
        samples = []
        with open(path, newline='') as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader)  # skip header
            desc = f'loading \'{path}\''
            for row in tqdm(reader, desc=desc):
                try:
                    guid = row[0]
                    sentence1 = row[3]
                    sentence2 = row[4]
                    label = row[5]
                    if self.valid_inputs(sentence1, sentence2, label):
                        label = int(label)
                        samples.append((sentence1, sentence2, label, guid))
                except:
                    pass
        return samples

class TwitterPPDBProcessor:
    """Data loader for TwittrPPDB."""

    def valid_inputs(self, sentence1, sentence2, label):
        return len(sentence1) > 0 and len(sentence2) > 0 and label != 3 
    
    def load_samples(self, path):
        samples = []
        with open(path, newline='') as f:
            reader = csv.reader(f, delimiter='\t')
            desc = f'loading \'{path}\''
            for row in tqdm(reader, desc=desc):
                try:
                    sentence1 = row[0]
                    sentence2 = row[1]
                    label = eval(row[2])[0]
                    if self.valid_inputs(sentence1, sentence2, label):
                        label = 0 if label < 3 else 1
                        samples.append((sentence1, sentence2, label, []))
                except:
                    pass
        return samples


class SWAGProcessor:
    """Data loader for SWAG."""

    def load_samples(self, path):
        samples = []
        with open(path, newline='') as f:
            reader = csv.reader(f, delimiter='\t')
            desc = f'loading \'{path}\''
            for row in tqdm(reader, desc=desc):
                try:
                    guid = row[5]
                    context = row[0]
                    start_ending = row[-1]
                    endings = row[1:5]
                    label = int(row[6])
                    samples.append((context, start_ending, endings, label, guid))
                except:
                    pass
        return samples


class HellaSWAGProcessor:
    """Data loader for HellaSWAG."""

    def load_samples(self, path):
        samples = []
        with open(path) as f:
            desc = f'loading \'{path}\''
            for line in f:
                try:
                    line = line.rstrip()
                    input_dict = json.loads(line)
                    context = input_dict['ctx_a']
                    start_ending = input_dict['ctx_b']
                    endings = input_dict['endings']
                    label = input_dict['label']
                    samples.append((context, start_ending, endings, label, []))
                except:
                    pass
        return samples


def select_processor():
    """Selects data processor using task name."""

    return globals()[f'{args.task}Processor']()


class TextDataset(Dataset):
    """
    Task-specific dataset wrapper. Used for storing, retrieving, encoding,
    caching, and batching samples.
    """

    def __init__(self, path, processor, threshold_sample=False):
        self.samples = processor.load_samples(path)
        self.cache = {}
        self.threshold_sample=threshold_sample

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        res = self.cache.get(i, None)
        if res is None:
            sample = self.samples[i]
            if args.task in ('SNLI', 'MNLI', 'QQP', 'MRPC', 'TwitterPPDB'):
                sentence1, sentence2, label, guid = sample
                input_ids, segment_ids, attention_mask = encode_pair_inputs(
                    sentence1, sentence2
                )
                packed_inputs = (sentence1, sentence2)
            elif args.task in ('SWAG', 'HellaSWAG'):
                context, ending_start, endings, label, guid = sample
                input_ids, segment_ids, attention_mask = encode_mc_inputs(
                    context, ending_start, endings
                )
            label_id = encode_label(label)
            res = ((input_ids, segment_ids, attention_mask), label_id, guid)
            self.cache[i] = res
        return res


class Model(nn.Module):
    """Pre-trained model for classification."""

    def __init__(self):
        super().__init__()
        self.model = AutoModel.from_pretrained(args.model)
        self.classifier = nn.Linear(768, n_classes)

    def forward(self, input_ids, segment_ids, attention_mask):
        # On SWAG and HellaSWAG, collapse the batch size and
        # choice size dimension to process everything at once
        if args.task in ('SWAG', 'HellaSWAG'):
            n_choices = input_ids.size(1)
            input_ids = input_ids.view(-1, input_ids.size(-1))
            segment_ids = segment_ids.view(-1, segment_ids.size(-1))
            attention_mask = attention_mask.view(-1, attention_mask.size(-1))
        transformer_params = {
            'input_ids': input_ids,
            'token_type_ids': (
                segment_ids if args.model == 'bert-base-uncased' else None
            ),
            'attention_mask': attention_mask,
        }
        transformer_outputs = self.model(**transformer_params)
        if args.task in ('SWAG', 'HellaSWAG'):
            pooled_output = transformer_outputs[1]
            logits = self.classifier(pooled_output)
            logits = logits.view(-1, n_choices)
        else:
            cls_output = transformer_outputs[0][:, 0]
            logits = self.classifier(cls_output)
        return logits




def smoothing_label(target, smoothing):
    '''
    Label smoothing. 
    '''
    _n_classes = n_classes if args.task not in ('SWAG', 'HellaSWAG') else 4
    confidence = 1. - smoothing
    smoothing_value = smoothing / (_n_classes - 1)
    one_hot = cuda(torch.full((_n_classes,), smoothing_value))
    model_prob = one_hot.repeat(target.size(0), 1)
    model_prob.scatter_(1, target.unsqueeze(1), confidence)
    return model_prob


def mixup(dataset1, dataset2,lam):
    '''
    MixUp operation for two given samples.
    '''
    input1, label1, guid1 = dataset1
    input2, label2, guid2 = dataset2
    output1 = model(*input1)
    output2 = model(*input2)
    if args.ls:
        if args.task == 'QQP':
            smoothing = 0.3
        elif args.task == 'SNLI':
            smoothing = 0.01
        else:
            smoothing = 0.3
        label1_onehot = smoothing_label(label1,smoothing)
        label2_onehot = smoothing_label(label2,smoothing)
    else:
        label1_onehot = F.one_hot(label1,num_classes=output1.shape[1])
        label2_onehot = F.one_hot(label2,num_classes=output2.shape[1])

    if output1.shape[0] != output2.shape[0]:
        min_idx = min(output1.shape[0], output2.shape[0])
        output1 = output1[:min_idx]
        output2 = output2[:min_idx]
        label1_onehot = label1_onehot[:min_idx]
        label2_onehot = label2_onehot[:min_idx]
        label1 = label1[:min_idx]
        label2 = label2[:min_idx]
    mixup_output = output1 * lam + output2 * (1-lam)
    mixup_label = label1_onehot * lam + label2_onehot * (1-lam)
    mixup_loss = torch.mean(torch.sum(-mixup_label * torch.log_softmax(mixup_output, dim=-1), dim=0))

    loss1 = criterion(output1,label1)
    loss2 = criterion(output2,label2)

    loss = 0.01*mixup_loss + 0.5*loss1 + 0.5*loss2
    return loss

def train(d1,d2=None,optimizer=None,scheduler=None,d1_ckpt=None, d2_ckpt=None):
    """Fine-tunes pre-trained model on training set."""

    model.train()
    train_loss = 0.
    if args.mixup:
        d1_loader = tqdm(load(d1, args.batch_size, True))
        d2_loader = tqdm(load(d2, args.batch_size, True))
        alpha = 0.4
        lam = np.random.beta(alpha,alpha)
        if len(d1) < len(d2):
            for i, (dataset1, dataset2) in enumerate(zip(cycle(d1_loader),d2_loader)):
                loss = mixup(dataset1,dataset2,lam)
                train_loss += loss.item()
                d1_loader.set_description(f"lr = {scheduler.get_lr()[0]:.8f}")
                d2_loader.set_description(f'total train loss = {(train_loss / (i+1)):.6f}')
                loss.backward()
                if args.max_grad_norm > 0.:
                    nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
        else:
            for i, (dataset1, dataset2) in enumerate(zip(d1_loader,cycle(d2_loader))):
                loss = mixup(dataset1,dataset2,lam)
                train_loss += loss.item()
                d1_loader.set_description(f"lr = {scheduler.get_lr()[0]:.9f}")
                d2_loader.set_description(f'total train loss = {(train_loss / (i+1)):.6f}')
                loss.backward()
                if args.max_grad_norm > 0.:
                    nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
        max_length = max(len(d1_loader), len(d2_loader))
        return train_loss/max_length
    else:
        train_loader = tqdm(load(d1, args.batch_size, True))
        grad_dict = {}
        for i, inputs in enumerate(train_loader):
            if args.threshold_sample:
                inputs, label, guid, original_sample_flag = inputs
            else:
                inputs, label, guid = inputs
            optimizer.zero_grad()
            output = model(*inputs)
            loss = criterion(output, label)
            train_loss += loss.item()
            train_loader.set_description(f'train loss = {(train_loss / (i+1)):.6f}')
            if args.grad_extract:
                output.retain_grad()
            loss.backward()
            if args.grad_extract:
                output_grad = output.grad.data.abs().tolist()
                for j in range(0,len(guid)):
                    grad_dict[guid[j]] = output_grad[j]
            if args.aum:
                records = aum_calculator.update(output,label,guid)
            if args.threshold_sample:
                for j in range(0,len(original_sample_flag)):
                    if original_sample_flag[j]:
                        aum_calculator_original.update(output[j].unsqueeze(0),label[j].unsqueeze(0),(guid[j]))
                    else:
                        aum_calculator_th.update(output[j].unsqueeze(0),label[j].unsqueeze(0),(guid[j]))
            if args.max_grad_norm > 0.:
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
        return train_loss / len(train_loader), grad_dict
        

def evaluate(dataset):
    """Evaluates pre-trained model on development set."""

    model.eval()
    eval_loss = 0.
    eval_loader = tqdm(load(dataset, args.batch_size, False))
    for i, inputs in enumerate(eval_loader):
        inputs, label, guid = inputs
        with torch.no_grad():
            loss = criterion(model(*inputs), label)
        eval_loss += loss.item()
        if i==0:
            pass
        else:
            eval_loader.set_description(f'eval loss = {(eval_loss / i):.6f}')
    return eval_loss / len(eval_loader)


model = cuda(Model())
processor = select_processor()
tokenizer = AutoTokenizer.from_pretrained(args.model)
criterion = nn.CrossEntropyLoss()

if args.mixup:
    if args.mixup_type == 'hard_easy':
        d1 = TextDataset(args.hard_train_path,processor)
        d2 = TextDataset(args.easy_train_path,processor)
        print(f'Hard-to-learn train samples = {len(d1)}')
        print(f'Easy-to-learn train samples = {len(d2)}')
    elif args.mixup_type == 'hard_ambig':
        d1 = TextDataset(args.hard_train_path,processor)
        d2 = TextDataset(args.ambig_train_path,processor)
        print(f'Hard-to-learn train samples = {len(d1)}')
        print(f'Ambiguous train samples = {len(d2)}')
    elif args.mixup_type == 'easy_ambig':
        d1 = TextDataset(args.easy_train_path,processor)
        d2 = TextDataset(args.ambig_train_path,processor)
        print(f'Easy-to-learn train samples = {len(d1)}')
        print(f'Ambiguous train samples = {len(d2)}')

else:
    if args.train_path:
        d1 = TextDataset(args.train_path, processor)
        print(f'train samples = {len(d1)}')
        if args.threshold_sample:
            if 'ambig' in args.train_path:
                data_type='ambig'
            elif 'easy' in args.train_path:
                data_type='easy'
            elif 'hard' in args.train_path:
                data_type='hard'
            else:
                data_type=''
            original_path = './calibration_data/'+args.task+'/original_'+data_type+'_sample.pkl'
            threshold_path = './calibration_data/'+args.task+'/threshold_'+data_type+'_sample.pkl'
            if os.path.exists(original_path) and os.path.exists(threshold_path):
                print("loading saved orignal and threshold instances...")
                d1_file = open(original_path,"rb")
                d1 = pickle.load(d1_file)
                d2_file = open(threshold_path,"rb")
                d2 = pickle.load(d2_file)
            else:
                label_dict = dict()
                with tqdm(len(d1), desc='processing threshold samples...') as pbar:
                    for d in d1:
                        label = int(d[-2].data.tolist())
                        if label not in label_dict:
                            label_dict[label] = [d]
                        else:
                            label_dict[label].append(d)
                        pbar.update(1)
                d2 = []
                d1_refine = []
                threshold_sample_length = int(len(d1)/(len(label_dict)+1))
                each_class_threshold_length = int(threshold_sample_length/len(label_dict))
                print("# of total instances, : ", str(len(d1)))
                print("Among them, the # of threshold instances : ", str(threshold_sample_length))
                print("We will asign ", str(each_class_threshold_length)," this amount of instances as a threshold sample per classes")
                new_class = max(label_dict.keys()) + 1
                print("New class label ", str(new_class))
                for key in label_dict:
                    print("Label ", str(key))
                    print("# of instance in this label ", str(len(label_dict[key])))
                    original_guids = []
                    if args.task not in ('SWAG', 'HellaSWAG'):
                        
                        try:
                            d1_refine += random.sample(label_dict[key],len(label_dict[key])-each_class_threshold_length)
                        except:
                            d1_refine += random.sample(label_dict[key], int(len(label_dict[key])*0.1))
                        for item in d1_refine:
                            _,_,guid = item
                            original_guids.append(guid)
                    else:
                        for item in random.sample(label_dict[key],len(label_dict[key])-each_class_threshold_length):
                            inputs, label, guid = item
                            input_ids = torch.cat([inputs[0],inputs[0][label.data.tolist()].unsqueeze(0)],dim=0)
                            segment_ids = torch.cat([inputs[1],inputs[1][label.data.tolist()].unsqueeze(0)],dim=0)
                            attention_mask = torch.cat([inputs[2],inputs[2][label.data.tolist()].unsqueeze(0)],dim=0)
                            d1_refine.append(((input_ids,segment_ids,attention_mask),encode_label(label),guid))
                            original_guids.append(guid)
                        
                    print("Original sample # ", str(len(d1_refine)))
                    for item in label_dict[key]:
                        if args.task not in ('SWAG','HellaSWAG'):
                            _,_,guid = item
                            if guid not in original_guids:
                                new_label = item[-2]
                                new_label += 1
                                d2.append((item[0],encode_label(new_class),item[-1]))
                            
                        else:
                            inputs, label, guid = item
                            if guid not in original_guids:
                                new_label = label.data.tolist() + 1
                                if label.data.tolist() == 3:
                                    fetch_idx = 0
                                else:
                                    fetch_idx = new_label
                                input_ids = torch.cat([inputs[0],inputs[0][fetch_idx].unsqueeze(0)],dim=0)
                                segment_ids = torch.cat([inputs[1],inputs[1][fetch_idx].unsqueeze(0)],dim=0)
                                attention_mask = torch.cat([inputs[2],inputs[2][fetch_idx].unsqueeze(0)],dim=0)
                                d2.append(((input_ids,segment_ids,attention_mask),encode_label(new_label),guid))
                    print("Accumulated Threshold sample # ", len(d2))
                print("Saving samples...")
                file = open(original_path,"wb")
                pickle.dump(d1_refine,file)
                print("Saving threshold samples...")
                th_file = open(threshold_path,"wb")
                pickle.dump(d2,th_file)
                d1 = d1_refine
            original_id, threshold_id = [], []
            d = []
            for item in d1:
                i1,i2,i3 = item
                d.append((i1,i2,i3,1))
                original_id.append(i3)
            for item in d2:
                i1,i2,i3 = item
                d.append((i1,i2,i3,0))
                threshold_id.append(i3)
            for item in threshold_id:
                if item in original_id:
                    print("There are duplicated items between threshold samples and original samples, Please check dataset!")
                    exit()
            print(f'train original samples = {len(d1)}')
            print(f'train threshold samples = {len(d2)}')
            print(f'train samples = {len(d)}')
            d1 = d

if args.dev_path:
    dev_dataset = TextDataset(args.dev_path, processor)
    print(f'dev samples = {len(dev_dataset)}')
if args.test_path:
    test_dataset = TextDataset(args.test_path, processor)
    print(f'test samples = {len(test_dataset)}')


if args.do_train:
    print()
    print('*** training ***')
    best_loss = float('inf')
    best_grad_dict = {}
    train_loader_length = len(load(d1,args.batch_size,True))
    t_total = train_loader_length // args.gradient_accumulation_steps * args.epochs
    optimizer = AdamW(adamw_params(model), lr=args.learning_rate, eps=1e-8)
    if args.mixup:
        scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
    for epoch in range(1, args.epochs + 1):
        if args.mixup:
            train_loss = train(d1=d1, d2=d2, optimizer=optimizer, scheduler=scheduler)
        else:
            train_loss, grad_dict = train(d1=d1, d2=None, optimizer=optimizer, scheduler=None)
        eval_loss = evaluate(dev_dataset)
        if eval_loss < best_loss:
            if args.grad_extract:
                best_grad_dict = grad_dict
            best_loss = eval_loss
            torch.save(model.state_dict(), args.ckpt_path)
            if args.grad_extract:
                grad_path = args.ckpt_path.replace(".pt",".pkl")
                pickle.dump(best_grad_dict,grad_path)
        print(
            f'epoch = {epoch} | '
            f'train loss = {train_loss:.6f} | '
            f'eval loss = {eval_loss:.6f}'
        )
    if args.threshold_sample:
        aum_calculator_original.finalize()
        aum_calculator_th.finalize()
        
    if args.aum:
        aum_calculator.finalize()

if args.do_evaluate:
    if not os.path.exists(args.ckpt_path):
        raise RuntimeError(f'\'{args.ckpt_path}\' does not exist')
    
    print()
    print('*** evaluating ***')

    output_dicts = []
    model.load_state_dict(torch.load(args.ckpt_path))
    model.eval()
    test_loader = tqdm(load(test_dataset, args.batch_size, False))

    for i, dataset1 in enumerate(test_loader):
        with torch.no_grad():
            inputs, label, guid = dataset1
            logits = model(*inputs)
            for j in range(logits.size(0)):
                probs = F.softmax(logits[j], -1)
                output_dict = {
                    'index': args.batch_size * i + j,
                    'true': label[j].item(),
                    'pred': logits[j].argmax().item(),
                    'conf': probs.max().item(),
                    'logits': logits[j].cpu().numpy().tolist(),
                    'probs': probs.cpu().numpy().tolist(),
                }
                output_dicts.append(output_dict)

    print(f'writing outputs to \'{args.output_path}\'')

    with open(args.output_path, 'w+') as f:
        for i, output_dict in enumerate(output_dicts):
            output_dict_str = json.dumps(output_dict)
            f.write(f'{output_dict_str}\n')

    y_true = [output_dict['true'] for output_dict in output_dicts]
    y_pred = [output_dict['pred'] for output_dict in output_dicts]
    y_conf = [output_dict['conf'] for output_dict in output_dicts]

    accuracy = accuracy_score(y_true, y_pred) * 100.
    f1 = f1_score(y_true, y_pred, average='macro') * 100.
    confidence = np.mean(y_conf) * 100.

    results_dict = {
        'accuracy': accuracy_score(y_true, y_pred) * 100.,
        'macro-F1': f1_score(y_true, y_pred, average='macro') * 100.,
        'confidence': np.mean(y_conf) * 100.,
    }
    for k, v in results_dict.items():
        print(f'{k} = {v}')
