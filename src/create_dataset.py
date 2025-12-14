import sys
import os
import re
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm_notebook
from collections import defaultdict
from subprocess import check_call

import torch
import torch.nn as nn


def to_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


# construct a word2id mapping that automatically takes increment when new words are encountered
word2id = defaultdict(lambda: len(word2id))
UNK = word2id['<unk>']
PAD = word2id['<pad>']

# turn off the word2id - define a named function here to allow for pickling


def return_unk():
    return UNK


def get_length(x):

    return x.shape[1]-(np.sum(x, axis=-1) == 0).sum(1)


class MOSI:
    def __init__(self, config):
        if config.sdk_dir is None:
            print("SDK path is not specified! Please specify first in constants/paths.py")
            exit(0)
        else:
            sys.path.append(str(config.sdk_dir))

        DATA_PATH = str(config.dataset_dir)
        CACHE_PATH = DATA_PATH + '/embedding_and_mapping.pt'

        # If cached data if already exists
        try:
            self.train = load_pickle(DATA_PATH + '/train.pkl')
            self.dev = load_pickle(DATA_PATH + '/dev.pkl')
            self.test = load_pickle(DATA_PATH + '/test.pkl')
            self.pretrained_emb, self.word2id = None, None

        except:

            # create folders for storing the data
            if not os.path.exists(DATA_PATH):
                check_call(' '.join(['mkdir', '-p', DATA_PATH]), shell=True)

            # load pickle file for unaligned acoustic and visual source
            # pickle_filename = '../datasets/MOSI/mosi_data_noalign.pkl'
            # csv_filename = '../datasets/MOSI/MOSI-label.csv'

            #pickle_filename = 'Multimodal-Infomax-main/datasets/MOSI/mosi_data_noalign.pkl'
            #csv_filename = 'Multimodal-Infomax-main/datasets/MOSI/MOSI-label.csv'
            pickle_filename = 'datasets/MOSI/mosi_data_noalign.pkl'
            csv_filename = 'datasets/MOSI/MOSI-label.csv'

            with open(pickle_filename, 'rb') as f:
 
                d = pickle.load(f)

            # read csv file for label and text   004 :1 cid_id
            # vid 000'03bSnISJMiM'001:'03bSnISJMiM'002:'03bSnISJMiM'003:'03bSnISJMiM'004:'03bSnISJMiM'
            # text 001:'THERE IS SAD PART'
            df = pd.read_csv(csv_filename)
            text = df['text']  
            vid = df['video_id']  
            cid = df['clip_id']  
            train_split_noalign = d['train']  # 1284
            dev_split_noalign = d['valid']  # 229
            test_split_noalign = d['test']  # 686

            # a sentinel epsilon for safe division, without it we will replace illegal values with a constant
            EPS = 1e-6

            # place holders for the final train/dev/test dataset
            self.train = train = []
            self.dev = dev = []
            self.test = test = []
            self.word2id = word2id

            # define a regular expression to extract the video ID out of the keys
            pattern = re.compile('(.*)_(.*)')
            num_drop = 0  # a counter to count how many data points went into some processing issues

            if True:
                v = np.concatenate(
                    (train_split_noalign['vision'], dev_split_noalign['vision'], test_split_noalign['vision']), axis=0)
                vlens = get_length(v)

                a = np.concatenate(
                    (train_split_noalign['audio'], dev_split_noalign['audio'], test_split_noalign['audio']), axis=0)
                alens = get_length(a)
                
                label = np.concatenate(
                    (train_split_noalign['labels'], dev_split_noalign['labels'], test_split_noalign['labels']), axis=0)
                # label[label > 0] = 2
                # label[label < 0] = -2
                print(label.sort())
                L_V = v.shape[1]
                L_A = a.shape[1]

            all_id = np.concatenate(
                (train_split_noalign['id'], dev_split_noalign['id'], test_split_noalign['id']), axis=0)[:, 0]
            
            all_id_list = list(
                map(lambda x: x.decode('utf-8'), all_id.tolist()))

            train_size = len(train_split_noalign['id'])
            dev_size = len(dev_split_noalign['id'])
            test_size = len(test_split_noalign['id'])

            dev_start = train_size
            test_start = train_size + dev_size

            all_csv_id = [(vid[i], str(cid[i])) for i in range(len(vid))]

            for i, idd in enumerate(all_id_list):
                # get the video ID and the features out of the aligned dataset
                
                idd1, idd2 = re.search(pattern, idd).group(1, 2)

                # matching process
                try:
                    index = all_csv_id.index((idd1, idd2))
                except:
                    exit()
                """
                    Retrive noalign data from pickle file 
                """
                
                _words = text[index].split()
                _label = label[i].astype(np.float32)  
                _visual = v[i]  
                _acoustic = a[i]  
                _vlen = vlens[i]  
                _alen = alens[i]  
                _id = all_id[i]

                # remove nan values 
                _visual = np.nan_to_num(_visual)
                _acoustic = np.nan_to_num(_acoustic)

                # remove speech pause tokens - this is in general helpful
                # we should remove speech pauses and corresponding visual/acoustic features together
                # otherwise modalities would no longer be aligned
                actual_words = []
                words = []
                visual = []
                acoustic = []

                # For non-align setting
                # we also need to record sequence lengths
                """TODO: Add length counting for other datasets 
                """
                for word in _words:
                    actual_words.append(word)

                
                visual = _visual[L_V - _vlen:, :]
                acoustic = _acoustic[L_A - _alen:, :]

                # z-normalization per instance and remove nan/infs
                # visual = np.nan_to_num((visual - visual.mean(0, keepdims=True)) / (EPS + np.std(visual, axis=0, keepdims=True)))
                # acoustic = np.nan_to_num((acoustic - acoustic.mean(0, keepdims=True)) / (EPS + np.std(acoustic, axis=0, keepdims=True)))
                
                if i < dev_start:
                    train.append(
                        ((words, visual, acoustic, actual_words, _vlen, _alen), _label, idd))
                elif i >= dev_start and i < test_start:
                    dev.append(
                        ((words, visual, acoustic, actual_words, _vlen, _alen), _label, idd))
                elif i >= test_start:
                    test.append(
                        ((words, visual, acoustic, actual_words, _vlen, _alen), _label, idd))
                else:
                    print(
                        f"Found video that doesn't belong to any splits: {idd}")

            print(f"Total number of {num_drop} datapoints have been dropped.")
            print("Dataset split")
            print("Train Set: {}".format(len(train)))
            print("Validation Set: {}".format(len(dev)))
            print("Test Set: {}".format(len(test)))
            word2id.default_factory = return_unk

            # Save glove embeddings cache too
            # self.pretrained_emb = pretrained_emb = load_emb(word2id, config.word_emb_path)
            # torch.save((pretrained_emb, word2id), CACHE_PATH)

            # Save pickles
            to_pickle(train, DATA_PATH + '/train.pkl')
            to_pickle(dev, DATA_PATH + '/dev.pkl')
            to_pickle(test, DATA_PATH + '/test.pkl')

    def get_data(self, mode):
        if mode == "train":
            
            return self.train, self.word2id, None
        elif mode == "valid":
            return self.dev, self.word2id, None
        elif mode == "test":
            return self.test, self.word2id, None
        else:
            print("Mode is not set properly (train/dev/test)")
            exit()


class MOSEI:
    def __init__(self, config):

        if config.sdk_dir is None:
            print("SDK path is not specified! Please specify first in constants/paths.py")
            exit(0)
        else:
            sys.path.append(str(config.sdk_dir))

        DATA_PATH = str(config.dataset_dir)
        CACHE_PATH = DATA_PATH + '/embedding_and_mapping.pt'

        # If cached data if already exists
        try:
            self.train = load_pickle(DATA_PATH + '/trainsss.pkl')
            self.dev = load_pickle(DATA_PATH + '/dev.pkl')
            self.test = load_pickle(DATA_PATH + '/test.pkl')
            self.pretrained_emb, self.word2id = None, None

        except:
            # create folders for storing the data
            if not os.path.exists(DATA_PATH):
                check_call(' '.join(['mkdir', '-p', DATA_PATH]), shell=True)

            # first we align to words with averaging, collapse_function receives a list of functions
            # dataset.align(text_field, collapse_functions=[avg])
            # load pickle file for unaligned acoustic and visual source
            pickle_filename = DATA_PATH+'/mosei_senti_data_noalign.pkl'
            csv_filename = DATA_PATH+'/MOSEI-label.csv'

            with open(pickle_filename, 'rb') as f:
                d = pickle.load(f)

            # read csv file for label and text
            df = pd.read_csv(csv_filename)
            text = df['text']
            vid = df['video_id']
            cid = df['clip_id']

            train_split_noalign = d['train']
            dev_split_noalign = d['valid']
            test_split_noalign = d['test']

            # a sentinel epsilon for safe division, without it we will replace illegal values with a constant
            EPS = 1e-6

            # place holders for the final train/dev/test dataset
            self.train = train = []
            self.dev = dev = []
            self.test = test = []
            self.word2id = word2id

            # define a regular expression to extract the video ID out of the keys
            # pattern = re.compile('(.*)\[.*\]')
            pattern = re.compile('(.*)_([.*])')
            num_drop = 0  # a counter to count how many data points went into some processing issues

            v = np.concatenate(
                (train_split_noalign['vision'], dev_split_noalign['vision'], test_split_noalign['vision']), axis=0)
            vlens = get_length(v)

            a = np.concatenate(
                (train_split_noalign['audio'], dev_split_noalign['audio'], test_split_noalign['audio']), axis=0)
            alens = get_length(a)

            label = np.concatenate(
                (train_split_noalign['labels'], dev_split_noalign['labels'], test_split_noalign['labels']), axis=0)

            L_V = v.shape[1]
            L_A = a.shape[1]

            all_id = np.concatenate(
                (train_split_noalign['id'], dev_split_noalign['id'], test_split_noalign['id']), axis=0)[:, 0]
            all_id_list = all_id.tolist()

            train_size = len(train_split_noalign['id'])
            dev_size = len(dev_split_noalign['id'])
            test_size = len(test_split_noalign['id'])

            dev_start = train_size
            test_start = train_size + dev_size

            all_csv_id = [(vid[i], str(cid[i])) for i in range(len(vid))]

            for i, idd in enumerate(all_id_list):
                # get the video ID and the features out of the aligned dataset

                # matching process
                try:
                    index = i
                except:
                    import ipdb
                    ipdb.set_trace()

                _words = text[index].split()
                _label = label[i].astype(np.float32)
                _visual = v[i]
                _acoustic = a[i]
                _vlen = vlens[i]
                _alen = alens[i]
                _id = '{}[{}]'.format(all_csv_id[0], all_csv_id[1])

                # remove nan values
                # label = np.nan_to_num(label)
                _visual = np.nan_to_num(_visual)
                _acoustic = np.nan_to_num(_acoustic)

                # remove speech pause tokens - this is in general helpful
                # we should remove speech pauses and corresponding visual/acoustic features together
                # otherwise modalities would no longer be aligned
                actual_words = []
                words = []
                visual = []
                acoustic = []

                for word in _words:
                    actual_words.append(word)

                visual = _visual[L_V - _vlen:, :]
                acoustic = _acoustic[L_A - _alen:, :]

                if i < dev_start:
                    train.append(
                        ((words, visual, acoustic, actual_words, _vlen, _alen), _label, idd))
                elif i >= dev_start and i < test_start:
                    dev.append(
                        ((words, visual, acoustic, actual_words, _vlen, _alen), _label, idd))
                elif i >= test_start:
                    test.append(
                        ((words, visual, acoustic, actual_words, _vlen, _alen), _label, idd))
                else:
                    print(
                        f"Found video that doesn't belong to any splits: {idd}")

            # print(f"Total number of {num_drop} datapoints have been dropped.")
            print(f"Total number of {num_drop} datapoints have been dropped.")
            print("Dataset split")
            print("Train Set: {}".format(len(train)))
            print("Validation Set: {}".format(len(dev)))
            print("Test Set: {}".format(len(test)))
            word2id.default_factory = return_unk

            # Save glove embeddings cache too
            # self.pretrained_emb = pretrained_emb = load_emb(word2id, config.word_emb_path)
            # torch.save((pretrained_emb, word2id), CACHE_PATH)
            self.pretrained_emb = None

            
            to_pickle(train, DATA_PATH + '/trainsss.pkl')
            to_pickle(dev, DATA_PATH + '/dev.pkl')
            to_pickle(test, DATA_PATH + '/test.pkl')

    def get_data(self, mode):

        if mode == "train":
            return self.train, self.word2id, self.pretrained_emb
        elif mode == "valid":
            return self.dev, self.word2id, self.pretrained_emb
        elif mode == "test":
            return self.test, self.word2id, self.pretrained_emb
        else:
            print("Mode is not set properly (train/dev/test)")
            exit()

class SIMS:
    def __init__(self, config):
        # 1. 路径检查
        if config.dataset_dir is None:
            print("Dataset path is not specified!")
            exit(0)
        
        DATA_PATH = str(config.dataset_dir)
        # 假设你下载的文件名是 ch_sims.pkl 或者 MMSA 处理好的文件名
        # 这里假设文件名是 'CH-SIMS_aligned.pkl' 或 similar，请根据实际情况修改
        # 建议把 .pkl 文件重命名为 sims.pkl 放在 dataset_dir 下
        CACHE_PATH = os.path.join(DATA_PATH, 'sims.pkl') 

        # 2. 加载数据
        # SIMS 数据通常比较小，我们可以直接加载，或者像 MOSI 一样做缓存逻辑
        # 这里为了保持一致性，我们尝试加载处理好的缓存，如果没有则处理原始文件
        try:
            self.train = load_pickle(os.path.join(DATA_PATH, 'train_cache.pkl'))
            self.dev = load_pickle(os.path.join(DATA_PATH, 'dev_cache.pkl'))
            self.test = load_pickle(os.path.join(DATA_PATH, 'test_cache.pkl'))
            print(f"Loaded Cached SIMS Data from {DATA_PATH}")
        except:
            print(f"Cache not found, loading raw SIMS pickle from {CACHE_PATH}...")
            
            # 加载 MMSA 格式的 pickle 文件
            with open(CACHE_PATH, 'rb') as f:
                data = pickle.load(f)
            
            # MMSA 格式通常是: data['train']['vision'], data['train']['raw_text']...
            self.train = self.reform_data(data['train'])
            self.dev = self.reform_data(data['valid']) # 注意 MMSA key 可能是 'valid'
            self.test = self.reform_data(data['test'])

            # 保存缓存，下次直接读取
            to_pickle(self.train, os.path.join(DATA_PATH, 'train_cache.pkl'))
            to_pickle(self.dev, os.path.join(DATA_PATH, 'dev_cache.pkl'))
            to_pickle(self.test, os.path.join(DATA_PATH, 'test_cache.pkl'))
            print("SIMS Data processed and cached.")

        # 为了兼容接口，返回空的 word2id (因为我们用 BERT)
        self.word2id = defaultdict(lambda: len(self.word2id))

    def reform_data(self, dataset):
        """
        将 MMSA 字典格式转换为 L-HDCL 需要的 Tuple 格式
        目标格式: ((words_idx, visual, acoustic, raw_text_list, v_len, a_len), label, id)
        """
        results = []
        
        # 【修改点 1】自动寻找正确的 Label Key
        # MMSA 通常使用 'regression_labels'，但也可能用 'labels' 或 'label'
        if 'regression_labels' in dataset:
            all_labels = dataset['regression_labels']
        elif 'labels' in dataset:
            all_labels = dataset['labels']
        elif 'label' in dataset:
            all_labels = dataset['label']
        else:
            print("Error: 找不到标签数据 (Keys: regression_labels, labels, label 都不存在)")
            print(f"Available keys: {dataset.keys()}")
            exit()
            
        num_samples = len(all_labels)
        
        # 【修改点 2】自动寻找 ID Key (防止报错)
        # 如果没有 id 键，就生成假的 id
        all_ids = dataset.get('id', [f'sims_{i}' for i in range(num_samples)])

        for i in range(num_samples):
            # 1. 提取特征
            visual = dataset['vision'][i]   # Shape: (Seq_Len, V_Dim)
            acoustic = dataset['audio'][i]  # Shape: (Seq_Len, A_Dim)
            label = all_labels[i]           # Shape: (1,) 或 Scalar
            
            segment_id = all_ids[i]

            # 2. 文本处理 (raw_text)
            # 增加鲁棒性：有些 key 叫 'text' 有些叫 'raw_text'
            if 'raw_text' in dataset:
                raw_text = dataset['raw_text'][i]
            elif 'text' in dataset: # 有些版本叫 text
                raw_text = dataset['text'][i]
            else:
                raw_text = "未知文本"

            if isinstance(raw_text, str):
                # 去掉空格 (有些数据集是 "我 爱 你")
                raw_text = raw_text.replace(" ", "")
                raw_text_list = list(raw_text) 
            else:
                # 如果已经是 list
                raw_text_list = raw_text

            # 3. 长度计算
            v_len = get_length(visual[np.newaxis, ...])[0]
            a_len = get_length(acoustic[np.newaxis, ...])[0]
            
            if v_len == 0: v_len = visual.shape[0]
            if a_len == 0: a_len = acoustic.shape[0]

            # 4. 占位符 words
            words_idx = [0] * len(raw_text_list)

            # 5. 组装
            entry = (
                (words_idx, visual, acoustic, raw_text_list, v_len, a_len),
                label,
                segment_id
            )
            results.append(entry)
            
        return results

    def get_data(self, mode):
        if mode == "train":
            return self.train, self.word2id, None
        elif mode == "valid":
            return self.dev, self.word2id, None
        elif mode == "test":
            return self.test, self.word2id, None
        else:
            print("Mode is not set properly (train/valid/test)")
            exit()
