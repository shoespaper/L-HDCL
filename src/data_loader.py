import random
import numpy as np
from tqdm import tqdm_notebook
from collections import defaultdict

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import *

# 【修改点1】引入 SIMS
from .create_dataset import MOSI, MOSEI, SIMS, PAD, UNK

# 【修改点2】删除原本在这里的全局 MODEL_PATH 和 bert_tokenizer
# 因为我们要根据命令行参数动态加载，不能写死在这里

class MSADataset(Dataset):
    def __init__(self, config):
        self.config = config

        # Fetch dataset
        path_str = str(config.data_dir).lower()
        if "mosi" in path_str:
            dataset = MOSI(config)
        elif "mosei" in path_str:
            dataset = MOSEI(config)
        # 【修改点3】添加 SIMS 支持
        elif "sims" in path_str:
            dataset = SIMS(config)
        else:
            print(f"Dataset not defined correctly: {config.data_dir}")
            exit()
            
        # 返回对齐的数据
        self.data, self.word2id, _ = dataset.get_data(config.mode)
        self.len = len(self.data)

        # 这一步是为了让 main.py 不报错
        config.word2id = self.word2id
        # config.pretrained_emb = self.pretrained_emb

    @property 
    def tva_dim(self):
        t_dim = 768
        # 动态获取维度
        return t_dim, self.data[0][0][1].shape[1], self.data[0][0][2].shape[1]

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


def get_loader(hp, config, shuffle=True):
    """Load DataLoader of given DialogDataset"""

    # 【修改点4】在函数内部初始化 Tokenizer，使用 hp.bert_name
    print(f"Initializing Tokenizer: {hp.bert_name}")
    bert_tokenizer = BertTokenizer.from_pretrained(hp.bert_name)

    dataset = MSADataset(config)

    print(config.mode)
    config.data_len = len(dataset)
    
    config.tva_dim = dataset.tva_dim

    if config.mode == 'train':
        hp.n_train = len(dataset)
    elif config.mode == 'valid':
        hp.n_valid = len(dataset)
    elif config.mode == 'test':
        hp.n_test = len(dataset)

    def collate_fn(batch):
        '''
        Collate functions assume batch = [Dataset[i] for i in index_set]
        '''  

        batch = sorted(batch, key=lambda x: len(x[0][3]), reverse=True)
        v_lens = []
        a_lens = []
        labels = []
        ids = []

        for sample in batch:
            if len(sample[0]) > 4:  # unaligned case / SIMS
                v_lens.append(torch.IntTensor([sample[0][4]]))  
                a_lens.append(torch.IntTensor([sample[0][5]]))  
            else:   # aligned cases
                v_lens.append(torch.IntTensor([len(sample[0][3])]))
                a_lens.append(torch.IntTensor([len(sample[0][3])]))
            
            # 兼容处理 label，防止 numpy/tensor 混用报错
            if isinstance(sample[1], torch.Tensor):
                labels.append(sample[1])
            else:
                # labels.append(torch.from_numpy(sample[1]))
                labels.append(torch.tensor(sample[1], dtype=torch.float32))
                
            ids.append(sample[2]) 
       
        vlens = torch.cat(v_lens)
        alens = torch.cat(a_lens)  
        
        # 堆叠 Labels
        if labels[0].dim() == 0:
            labels = torch.stack(labels)
        else:
            labels = torch.cat(labels, dim=0)

        # 统一维度 [Batch, 1]
        if labels.dim() == 1:
            labels = labels.unsqueeze(1)

        # MOSEI sentiment labels locate in the first column of sentiment matrix
        if labels.size(1) == 7:
            labels = labels[:, 0][:, None]

        # Rewrite this
        def pad_sequence(sequences, target_len=-1, batch_first=False, padding_value=0.0):
            if target_len < 0:
                max_size = sequences[0].size()
                trailing_dims = max_size[1:]
            else:
                max_size = target_len
                trailing_dims = sequences[0].size()[1:]

            max_len = max([s.size(0) for s in sequences])
            if batch_first:
                out_dims = (len(sequences), max_len) + trailing_dims
            else:  
                out_dims = (max_len, len(sequences)) + trailing_dims
            out_tensor = sequences[0].new_full(out_dims, padding_value)
            for i, tensor in enumerate(sequences):
                length = tensor.size(0)
                # use index notation to prevent duplicate references to the tensor
                if batch_first:  
                    out_tensor[i, :length, ...] = tensor
                else:
                    out_tensor[:length, i, ...] = tensor
            return out_tensor

        # Glove padding (保留逻辑以防万一，填0即可)
        sentences = pad_sequence([torch.LongTensor(sample[0][0])
                                  for sample in batch], padding_value=PAD)

        visual = pad_sequence([torch.FloatTensor(sample[0][1])
                               for sample in batch], target_len=vlens.max().item())
        acoustic = pad_sequence([torch.FloatTensor(sample[0][2])
                                 for sample in batch], target_len=alens.max().item())

        # BERT-based features input prep
        SENT_LEN = 50
        
        bert_details = []
        for sample in batch:
            raw_text_list = sample[0][3]
            
            # 【修改点5】中文文本处理
            # 你的原始代码是 " ".join，这会在中文里加空格，这里做一个智能判断
            if 'chinese' in hp.bert_name.lower() or 'macbert' in hp.bert_name.lower():
                text = "".join(raw_text_list) # 中文不加空格
            else:
                text = " ".join(raw_text_list) # 英文加空格

            encoded_bert_sent = bert_tokenizer.encode_plus(
                text, max_length=SENT_LEN, add_special_tokens=True, truncation=True, padding='max_length')
            bert_details.append(encoded_bert_sent)

        # Bert things are batch_first 
        bert_sentences = torch.LongTensor(
            [sample["input_ids"] for sample in bert_details])
        bert_sentence_types = torch.LongTensor(
            [sample["token_type_ids"] for sample in bert_details])
        bert_sentence_att_mask = torch.LongTensor(
            [sample["attention_mask"] for sample in bert_details])

        # lengths are useful later in using RNNs
        lengths = torch.LongTensor([len(sample[0][0]) for sample in batch])
        if (vlens <= 0).sum() > 0:
            vlens[vlens <= 0] = 1 # 修复 numpy 写法 warning

        return sentences, visual, vlens, acoustic, alens, labels, lengths, bert_sentences, bert_sentence_types, bert_sentence_att_mask, ids
   
    # 【修改点6：关键修复】
    # 删除了 generator=torch.Generator(device='cuda')
    # 因为 main.py 里的 global cuda default 和 DataLoader 的 CPU shuffle 逻辑冲突
    # 删除后，PyTorch 会使用默认的 CPU 生成器，这样就兼容了
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,   
        collate_fn=collate_fn,
        num_workers=0 # 建议设为0，减少多进程带来的 device 问题
    )  

    return data_loader