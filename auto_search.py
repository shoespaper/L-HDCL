import os
import subprocess
import sys
import glob
import pandas as pd
import time
import numpy as np

# ================= 1. 实验配置区域 (在这里修改你的参数) =================

# 你想搜索的权重列表
WEIGHTS_TO_SEARCH = [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1]
#WEIGHTS_TO_SEARCH = np.arange(0.00, 0.1 + 0.01, 0.01).tolist() # 从 0.00 到 0.10，步长 0.01
# 基础参数配置 (你想记录在 CSV 里的参数都写在这里)
# 脚本会自动将这些参数拼接成命令行，并记录到结果表中
BASE_CONFIG = {
    "dataset": "mosei",
    #"common_dim": 256,       # 你想调整的维度
    #"layers": 2,             # Transformer 层数
    "batch_size": 128,        # Batch Size
    #"num_epochs": 50,        # 训练轮数
    "lr_main": 8e-5,         # 学习率
    "clip": 5.0,              # 梯度裁剪阈值
    "common_dim": 128,       # 公共空间维度 
    #"dropout_prj": 0.5,    # 投影层 Dropout 比例
    #"attn_dropout": 0.2,   # 注意力 Dropout 比例

}

# 结果保存文件名
SUMMARY_FILE = "tuning_summary.csv"

# Python解释器
PYTHON_EXE = sys.executable 
# ======================================================================

def get_latest_log_csv(dataset_name):
    """找到 logs/dataset/ 目录下最新的 metrics_xxx.csv"""
    log_dir = os.path.join("logs", dataset_name)
    
    if not os.path.exists(log_dir):
        return None

    # 获取所有 metrics_*.csv 文件
    list_of_files = glob.glob(os.path.join(log_dir, "metrics_*.csv")) 
    
    if not list_of_files:
        return None
        
    # 按创建时间排序，找最新的
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file

def analyze_log_file(csv_path):
    """读取单次训练的 CSV，找到 Acc-2 最高的那一行"""
    try:
        # 尝试读取，如果文件为空或格式错误，Pandas 会报错
        df = pd.read_csv(csv_path)
        
        if df.empty or 'Test_Acc_2' not in df.columns:
            print(f"⚠️ 日志文件 {csv_path} 为空或缺少关键列。")
            return None
            
        # 找到 Test_Acc_2 最大值对应的索引
        best_idx = df['Test_Acc_2'].idxmax()
        best_row = df.iloc[best_idx]
        
        return {
            "Best_Epoch": int(best_row['Epoch']),
            "Best_Acc2": best_row['Test_Acc_2'],
            "Best_F1": best_row['Test_F1'],
            "MAE": best_row['Test_MAE'],
            "Corr": best_row['Test_Corr']
        }
    except pd.errors.EmptyDataError:
        print(f"⚠️ 日志文件 {csv_path} 是空的。")
        return None
    except Exception as e:
        print(f"❌ 读取日志文件出错 {csv_path}: {e}")
        return None

def main():
    # 1. 准备 CSV 表头
    config_keys = list(BASE_CONFIG.keys())
    result_keys = ["Best_Acc2", "Best_F1", "MAE", "Corr", "Best_Epoch"]
    # 强制定义列的顺序
    headers = ["Timestamp", "Aux_Weight"] + config_keys + result_keys + ["Log_File"]

    # 初始化汇总文件
    if not os.path.exists(SUMMARY_FILE):
        pd.DataFrame(columns=headers).to_csv(SUMMARY_FILE, index=False)

    print(f"🚀 开始自动实验。共 {len(WEIGHTS_TO_SEARCH)} 组参数。")
    print(f"📂 结果将保存在: {SUMMARY_FILE}\n")

    for weight in WEIGHTS_TO_SEARCH:
        print(f"==================================================")
        print(f"▶ 正在运行: Aux_Weight = {weight}, Config = {BASE_CONFIG}")
        print(f"==================================================")
        
        # 2. 动态构造命令
        # 基础命令
        cmd = [PYTHON_EXE, "-m", "src.main", "--aux_weight", str(weight)]
        
        # 将 BASE_CONFIG 里的键值对拼接到命令中
        # 例如: --common_dim 256
        for key, value in BASE_CONFIG.items():
            cmd.append(f"--{key}")
            cmd.append(str(value))
        
        # 打印完整命令供检查
        print(f"执行命令: {' '.join(cmd)}")

        # 3. 运行训练
        try:
            # check=True 确保如果 python main.py 报错，脚本能捕获并提示
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError:
            print(f"❌ 实验 (Weight={weight}) 运行失败（Exit Code != 0）！跳过...")
            continue
        except KeyboardInterrupt:
            print("\n⛔ 用户手动中断。")
            break

        # 4. 获取日志
        latest_csv = get_latest_log_csv(BASE_CONFIG['dataset'])
        if not latest_csv:
            print("⚠️ 未找到日志文件（可能是训练未启动或目录被删），跳过记录。")
            continue

        # 5. 分析结果
        result = analyze_log_file(latest_csv)
        if result:
            print(f"✅ 实验成功! Best Acc: {result['Best_Acc2']:.4f}")
            
            # 6. 构造汇总行数据
            row_data = {
                "Timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                "Aux_Weight": weight,
                **BASE_CONFIG, 
                **result,      
                "Log_File": latest_csv
            }
            
            # ================= [关键修改在这里] =================
            # 创建 DataFrame 时，显式传入 columns=headers
            # 这样 Pandas 就会强制按照 headers 的顺序排列数据，绝不会错位
            df_row = pd.DataFrame([row_data], columns=headers)
            
            df_row.to_csv(SUMMARY_FILE, mode='a', header=False, index=False)
            # ==================================================
        else:
            print("⚠️ 无法从日志中提取有效结果。")
            
        time.sleep(1)

    print(f"\n🎉 所有实验结束！请打开 {SUMMARY_FILE} 查看对比结果。")

if __name__ == "__main__":
    main()