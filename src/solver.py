import torch
from torch import nn
import os
import csv
import sys
import torch.optim as optim
import numpy as np
import time
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score
from .utils.eval_metrics import *
from .utils.tools import *
from .model import MMIM

class Solver(object):
    def __init__(self, hyp_params, train_loader, dev_loader, test_loader, is_train=True, model=None, pretrained_emb=None):
        self.hp = hp = hyp_params
        self.epoch_i = 0
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader
        self.is_train = is_train
        self.model = model
        
        # 这些参数在新的动态权重机制下可能不再直接使用，但为了兼容性先留着
        self.alpha = hp.alpha
        self.beta = hp.beta
        self.nce2 = hp.nce2
        self.nce3 = hp.nce3
        
        self.y_true = torch.tensor([])
        self.y_pre = torch.tensor([])
        self.p_value = []
        self.t_statistic = []
        self.update_batch = hp.update_batch

        if model is None:
            self.model = model = MMIM(hp)

        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            model = model.to(self.device)
        else:
            self.device = torch.device("cpu")

        # Criterion
        if self.hp.dataset == "ur_funny":
            self.criterion = criterion = nn.CrossEntropyLoss(reduction="mean")
        else: # mosi and mosei are regression datasets
            self.criterion = criterion = nn.L1Loss(reduction="mean")

        # Optimizer
        self.optimizer = {}
        if self.is_train:
            mmilb_param = []
            main_param = []
            bert_param = []
            
            for name, p in model.named_parameters():
                if p.requires_grad:
                    if 'bert' in name:
                        bert_param.append(p)
                    elif 'mi' in name:
                        mmilb_param.append(p)
                    else:
                        main_param.append(p)
            
            # Xavier Initialization
            for p in (mmilb_param + main_param):
                if p.dim() > 1:
                    nn.init.xavier_normal_(p)

            optimizer_main_group = [
                {'params': bert_param, 'weight_decay': hp.weight_decay_bert, 'lr': hp.lr_bert},
                {'params': main_param, 'weight_decay': hp.weight_decay_main, 'lr': hp.lr_main}
            ]

            self.optimizer_main = getattr(torch.optim, self.hp.optim)(
                optimizer_main_group
            )

        self.scheduler_main = ReduceLROnPlateau(
            self.optimizer_main, mode='min', patience=hp.when, factor=0.5, verbose=True)

        # ============================================================
        # 【新增】 日志保存初始化
        # ============================================================
        if self.is_train:
            # 1. 创建日志目录 logs/dataset_name/
            self.log_dir = os.path.join('logs', self.hp.dataset)
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)

            # 2. 生成带时间戳的文件名 (防止覆盖)
            now_time = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
            self.log_txt_path = os.path.join(self.log_dir, f'log_{now_time}.txt')
            self.log_csv_path = os.path.join(self.log_dir, f'metrics_{now_time}.csv')

            print(f"Training logs will be saved to: {self.log_dir}")

            # 3. 初始化 CSV 文件头 (根据你需要记录的指标修改)
            with open(self.log_csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'Epoch', 
                    'Train_Loss', 'Valid_Loss', 'Test_Loss', 
                    'Test_MAE', 'Test_Corr', 'Test_Acc_2', 'Test_F1',
                    'Best_Valid_Loss'
                ])
            
            # 4. 把当前超参数写入 TXT 日志开头
            self.print_log(f"Hyperparameters: {self.hp}")
            #self.print_log(f"aux_weight: {self.hp.aux_weight:.4f}")

    # ============================================================
    # 【新增】 辅助函数：同时打印到控制台和写入 TXT
    # ============================================================
    def print_log(self, msg):
        print(msg)  # 打印到控制台
        if self.is_train:
            with open(self.log_txt_path, 'a') as f:
                f.write(msg + '\n')  # 写入文件

    def train_and_eval(self):
        model = self.model
        optimizer_main = self.optimizer_main
        scheduler_main = self.scheduler_main
        criterion = self.criterion

        def train(model, optimizer, criterion, epochs, stage=1):
            epoch_loss = 0
            model.train()
            num_batches = self.hp.n_train // self.hp.batch_size
            proc_loss, proc_size = 0, 0
            start_time = time.time()
            left_batch = self.update_batch

            for i_batch, batch_data in enumerate(self.train_loader):
                text, visual, vlens, audio, alens, y, l, bert_sent, bert_sent_type, bert_sent_mask, ids = batch_data
                model.zero_grad()
                
                # 数据移至 GPU
                dd = self.device
                with torch.cuda.device(dd): # 或者直接用 .to(self.device)
                    text, visual, audio, y, l, bert_sent, bert_sent_type, bert_sent_mask = \
                        text.to(dd), visual.to(dd), audio.to(dd), y.to(dd), l.to(dd), bert_sent.to(dd), \
                        bert_sent_type.to(dd), bert_sent_mask.to(dd)
                    if self.hp.dataset == "ur_funny":
                        y = y.squeeze()

                batch_size = y.size(0)

                # =================================================================
                # 【修改点 1】 解包模型的新输出
                # 对应 model.py 的 return: preds, loss_diff, loss_l1, loss_l2, loss_l3, log_vars
                # =================================================================
                preds, loss_diff, loss_l1, loss_l2, loss_l3, loss_log_vars = \
                    model(text, visual, audio, vlens, alens, bert_sent, bert_sent_type, bert_sent_mask, y)

                # =================================================================
                # 【修改点 2】 NaN 检查逻辑更新
                # =================================================================
                if torch.isnan(preds).any() or torch.isinf(preds).any():
                    self.print_log(f"!!! FATAL: 'preds' is NaN/Inf at batch {i_batch}. Stop.")
                    sys.exit()
                if torch.isnan(loss_l3).any() or torch.isinf(loss_l3).any():
                    self.print_log(f"!!! FATAL: 'Level-3 Loss' is NaN/Inf at batch {i_batch}. Stop.")
                    sys.exit()

                # =================================================================
                # 【修改点 3】 计算总 Loss (动态权重)
                # =================================================================
                # 计算任务损失 (MAE/CrossEntropy)
                task_loss = criterion(preds, y)

                if self.hp.contrast: # 如果开启对比学习
                    # 使用同方差不确定性加权 (Homoscedastic Uncertainty Weighting)
                    # Loss = exp(-log_var) * loss + 0.5 * log_var
                    
                    # 索引对应关系 (根据 model.py init 里的顺序):
                    # 0: Task, 1: Diff, 2: L1, 3: L2, 4: L3
                    
                    w_task = torch.exp(-loss_log_vars[0]) * task_loss + 0.5 * loss_log_vars[0]
                    w_diff = torch.exp(-loss_log_vars[1]) * loss_diff + 0.5 * loss_log_vars[1]
                    w_l1   = torch.exp(-loss_log_vars[2]) * loss_l1   + 0.5 * loss_log_vars[2]
                    w_l2   = torch.exp(-loss_log_vars[3]) * loss_l2   + 0.5 * loss_log_vars[3]
                    w_l3   = torch.exp(-loss_log_vars[4]) * loss_l3   + 0.5 * loss_log_vars[4]
                    #TODO 强迫模型把 90% 的精力放在 task_loss (情感预测) 上
                    aux_weight = self.hp.aux_weight 
                    
                    #loss = w_task + self.hp.aux_weight * (w_diff + w_l1 + w_l2 + w_l3)
                    #loss = task_loss + aux_weight*(0.1 * loss_diff + 0.5 * (loss_l1 + loss_l2 + loss_l3))
                    loss = task_loss + aux_weight*(0.1 * loss_diff + 0.5 * (loss_l1 + loss_l2 + loss_l3))
                else:
                    loss = task_loss
                
                # 反向传播前最后检查
                if torch.isnan(loss) or torch.isinf(loss):
                    self.print_log(f"!!! FATAL: Total loss is NaN/Inf at batch {i_batch}.")
                    sys.exit()
                
                loss.backward()

                left_batch -= 1
                if left_batch == 0:
                    left_batch = self.update_batch
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.hp.clip)
                    optimizer.step()

                proc_loss += loss.item() * batch_size
                proc_size += batch_size
                epoch_loss += loss.item() * batch_size

                if i_batch % self.hp.log_interval == 0 and i_batch > 0:
                    avg_loss = proc_loss / proc_size
                    elapsed_time = time.time() - start_time
                    # 打印各个子 Loss 的数值，方便观察
                    
                    self.print_log('Epoch {:2d} | Batch {:3d}/{:3d} | Time {:5.2f}ms | Loss {:5.4f} | '
                          'Task {:.3f} Diff {:.3f} L1 {:.3f} L2 {:.3f} L3 {:.3f}'.format(
                          epoch, i_batch, num_batches, elapsed_time * 1000 / self.hp.log_interval,
                          avg_loss, task_loss.item(), loss_diff.item(), loss_l1.item(), loss_l2.item(), loss_l3.item()))
                    
                    proc_loss, proc_size = 0, 0
                    start_time = time.time()
                    
            return epoch_loss / self.hp.n_train

        def evaluate(model, criterion, epochs, test=False):
            model.eval()
            loader = self.test_loader if test else self.dev_loader
            total_loss = 0.0
            results = []
            truths = []

            with torch.no_grad():
                for batch in loader:
                    text, vision, vlens, audio, alens, y, lengths, bert_sent, bert_sent_type, bert_sent_mask, ids = batch
                    dd = self.device
                    with torch.cuda.device(dd):
                        text, audio, vision, y = text.to(dd), audio.to(dd), vision.to(dd), y.to(dd)
                        lengths = lengths.to(dd)
                        bert_sent, bert_sent_type, bert_sent_mask = bert_sent.to(dd), bert_sent_type.to(dd), bert_sent_mask.to(dd)
                        
                        if self.hp.dataset == 'iemocap': y = y.long()
                        if self.hp.dataset == 'ur_funny': y = y.squeeze()

                    batch_size = lengths.size(0)

                    # =================================================================
                    # 【修改点 4】 评估时只需取 preds
                    # =================================================================
                    # 解包所有返回，虽然只用 preds
                    preds, _, _, _, _, _ = model(
                        text, vision, audio, vlens, alens, bert_sent, bert_sent_type, bert_sent_mask)

                    if self.hp.dataset in ['mosi', 'mosei', 'mosei_senti'] and test:
                        eval_criterion = nn.L1Loss()
                        total_loss += eval_criterion(preds, y).item() * batch_size
                    else:
                        total_loss += criterion(preds, y).item() * batch_size

                    results.append(preds)
                    truths.append(y)

            avg_loss = total_loss / (self.hp.n_test if test else self.hp.n_valid)
            results = torch.cat(results)
            truths = torch.cat(truths)
            return avg_loss, results, truths

        # ------------------------------------------------------------------------
        # 主训练循环 (基本保持不变，只是调用上面修改过的 train/evaluate)
        # ------------------------------------------------------------------------
        patience = self.hp.patience
        best_valid = float('inf')
        best_mae = float('inf')
        best_epoch = -1
        
        for epoch in range(1, self.hp.num_epochs+1):
            start = time.time()
            train_loss = train(model, optimizer_main, criterion, epoch, 1)
            val_loss, _, _ = evaluate(model, criterion, epoch, test=False)
            test_loss, results, truths = evaluate(model, criterion, epoch, test=True)

            end = time.time()
            duration = end - start
            scheduler_main.step(val_loss)
            self.print_log("-" * 50)
            self.print_log('Epoch {:2d} | Time {:5.4f} sec | Valid Loss {:5.4f} | Test Loss {:5.4f}'.format(
                epoch, duration, val_loss, test_loss))
            self.print_log("-" * 50)

            # ============================================================
            # 【新增】 计算指标并写入 CSV (每轮都记，方便画图)
            # ============================================================
            # 这里我们需要临时计算一下指标，或者利用 eval_mosi 的返回值
            # 为了简单，我们这里简单调用 eval_metrics 里的计算函数，不打印，只获取值
            # 注意：这里假设数据集是 MOSI/MOSEI 回归任务
            mae = np.mean(np.abs(results.cpu().numpy() - truths.cpu().numpy()))
            corr = np.corrcoef(results.cpu().numpy().reshape(-1), truths.cpu().numpy().reshape(-1))[0][1]
            
            # 二分类准确率 (Non-negative)
            preds_binary = (results.cpu().numpy() >= 0)
            truths_binary = (truths.cpu().numpy() >= 0)
            acc_2 = accuracy_score(truths_binary, preds_binary)
            f1_2 = f1_score(truths_binary, preds_binary, average='weighted')

            # 写入 CSV
            with open(self.log_csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch, 
                    f"{train_loss:.4f}", 
                    f"{val_loss:.4f}", 
                    f"{test_loss:.4f}",
                    f"{mae:.4f}", 
                    f"{corr:.4f}", 
                    f"{acc_2:.4f}", 
                    f"{f1_2:.4f}",
                    f"{best_valid:.4f}"
                ])
            # ============================================================

            if val_loss < best_valid:
                patience = self.hp.patience
                best_valid = val_loss
                if self.hp.dataset == "ur_funny":
                    eval_humor(results, truths, True)
                elif test_loss < best_mae:
                    best_epoch = epoch
                    best_mae = test_loss
                    if self.hp.dataset in ["mosei_senti", "mosei"]:
                        eval_result= eval_mosei_senti(results, truths, True)
                    elif self.hp.dataset == 'mosi':
                        eval_result = eval_mosi(results, truths, True)
                    elif self.hp.dataset == 'iemocap':
                        eval_result = eval_iemocap(results, truths)

                    best_results = results
                    best_truths = truths
                    name=self.hp.dataset+'_best_model'
                    self.print_log(f"Saved model at pre_trained_models/{name}+时间戳.pt! " + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
                    save_model(self.hp, model,name=name, tag=self.hp.aux_weight)
                    # 可以在 TXT 里标记一下这是目前最好的
                    self.print_log(f"*** New Best Epoch: {epoch} (MAE: {best_mae:.4f}) ***")
                    self.print_log("{}\n".format(eval_result))
            else:
                patience -= 1
                if patience == 0:
                    break
        
        self.print_log(f'Best epoch: {best_epoch}')
        if self.hp.dataset in ["mosei_senti", "mosei"]:
            eval_mosei_senti(best_results, best_truths, True)
        elif self.hp.dataset == 'mosi':
            self.best_dict = eval_mosi(best_results, best_truths, True)
        elif self.hp.dataset == 'iemocap':
            eval_iemocap(results, truths)
        sys.stdout.flush()