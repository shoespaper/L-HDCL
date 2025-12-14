#-------------------------------------------------------------------------------------------------------------------#
#--------We use the open source baseline MMIM (https://github.com/declare-lab/Multimodal-Infomax?tab=readme-ov-file)#
#-------------------------------------------------------------------------------------------------------------------#
import os
import gc
import torch
import argparse
import numpy as np

from src.utils import *
from torch.utils.data import DataLoader
from src.solver import Solver
from src.config import get_args, get_config, output_dim_dict, criterion_dict
from src.data_loader import get_loader

# 建议把 CUDA 设置放在 main 函数里或由 args 控制，这里保持默认即可
# 如果你想指定显卡，建议在运行脚本时用 CUDA_VISIBLE_DEVICES=0 python main.py
# torch.cuda.set_device(0) 

def set_seed(seed):
    """
    设置随机种子以保证实验可复现性
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # 这一行经常导致 DataLoader 在 CPU 加载数据时报错
        torch.set_default_tensor_type('torch.cuda.FloatTensor') 

if __name__ == '__main__':
    args = get_args()
    dataset = str.lower(args.dataset.strip())

    print(f"Batch Size: {args.batch_size}")
    set_seed(args.seed)

    print("Start loading the data....")
    train_config = get_config(dataset, mode='train', batch_size=args.batch_size)
    valid_config = get_config(dataset, mode='valid', batch_size=args.batch_size)
    test_config = get_config(dataset, mode='test',  batch_size=args.batch_size)

    # 加载数据
    train_loader = get_loader(args, train_config, shuffle=True)
    print(f"Dataset: {args.dataset}")
    print('Training data loaded!')
    
    valid_loader = get_loader(args, valid_config, shuffle=False)
    print('Validation data loaded!')
    
    test_loader = get_loader(args, test_config, shuffle=False)
    print('Test data loaded!')
    print('Finish loading the data....')

    # 调试用的，如果遇到 NaN 可以解开注释查看反向传播哪里出了问题
    # torch.autograd.set_detect_anomaly(True)

    args.word2id = train_config.word2id

    # =================================================================
    # 【关键修改】 显式定义模型各模态的“输出维度”
    # model.py 的 DecouplingLayer 需要这些参数来初始化
    # =================================================================
    
    # 1. 获取输入维度 (tva_dim 在 config.py 里定义)
    # 通常是: 768 (BERT), 20 (Visual), 5 (Audio)
    args.d_tin, args.d_vin, args.d_ain = train_config.tva_dim

    # 2. 计算输出维度 (Out Dims)
    # 文本: BERT 输出维度通常不变
    args.d_tout = args.d_tin 

    # 音频 & 视觉: 取决于 LSTM 的 hidden size 和是否双向
    # args.d_ah, args.d_vh, args.bidirectional 这些参数在 get_args() 里默认有
    if args.bidirectional:
        args.d_aout = args.d_ah * 2
        args.d_vout = args.d_vh * 2
    else:
        args.d_aout = args.d_ah
        args.d_vout = args.d_vh
    
    print(f"Model Dims -> Text: {args.d_tout}, Audio: {args.d_aout}, Visual: {args.d_vout}")

    # =================================================================

    # 其他配置
    args.dataset = args.data = dataset
    args.when = args.when
    args.n_class = output_dim_dict.get(dataset, 1)
    args.criterion = criterion_dict.get(dataset, 'MSELoss')

    # 实例化 Solver 并开始训练
    solver = Solver(args, train_loader=train_loader, dev_loader=valid_loader,
                    test_loader=test_loader, is_train=True)
    
    solver.train_and_eval()