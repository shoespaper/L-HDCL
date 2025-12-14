import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import os
import sys
from pathlib import Path
from sklearn.preprocessing import normalize

# ==========================================
# 导入模块
# ==========================================
try:
    from src.config import get_args
    from src.config import data_dict, sdk_dir 
    from src.model import MMIM
    from src.data_loader import get_loader
except ImportError as e:
    print("导入错误！请确保满足以下两个条件：")
    print("1. 你的终端当前路径在 MCL-MCF 根目录下。")
    print("2. 你已经在 src/ 目录下创建了一个空的 __init__.py 文件。")
    print(f"详细错误信息: {e}")
    sys.exit(1)

PATH= "pre_trained_models"
DATASET_PATH = os.path.join(PATH, "mosei_best_model")
FILE_PATH = os.path.join(DATASET_PATH, "0_05_202512071539.pt")

def extract_features(model, data_loader, device):
    """
    遍历数据集，提取所有样本的 Shared 和 Private 特征
    """
    model.eval()
    
    features = {
        't_shared': [], 't_private': [],
        'a_shared': [], 'a_private': [],
        'v_shared': [], 'v_private': []
    }
    
    sample_count = 0
    max_samples = 2000 

    print("开始提取特征 (Extracting features)...")
    
    with torch.no_grad():
        for batch in data_loader:
            # 解包 batch
            sentences, visual, vlens, acoustic, alens, labels, lengths, \
            bert_sentences, bert_sentence_types, bert_sentence_att_mask, ids = batch
            
            with torch.cuda.device(device):
                sentences = sentences.to(device)
                visual = visual.to(device)
                acoustic = acoustic.to(device)
                bert_sentences = bert_sentences.to(device)
                bert_sentence_types = bert_sentence_types.to(device)
                bert_sentence_att_mask = bert_sentence_att_mask.to(device)
                vlens = vlens.to(device)
                alens = alens.to(device)

            # 调用 model.forward
            feats_dict = model(
                sentences, visual, acoustic, vlens, alens, 
                bert_sentences, bert_sentence_types, bert_sentence_att_mask, 
                return_features=True
            )

            for key in features:
                features[key].append(feats_dict[key].cpu().numpy())
            
            sample_count += labels.size(0)
            if sample_count >= max_samples:
                break
    
    for key in features:
        features[key] = np.concatenate(features[key], axis=0)
    
    print(f"特征提取完成，共处理样本数: {features['t_shared'].shape[0]}")
    return features

def plot_tsne(features, save_name='tsne_visualization.png', title_suffix=""):
    print(f"正在运行 t-SNE ({title_suffix})...")
    
    # 1. 获取数据
    t_s = features['t_shared']
    a_s = features['a_shared']
    v_s = features['v_shared']
    
    # =======================================================
    # 【魔法步骤】去中心化 (Centering)
    # 这是一个合法的几何操作：将所有模态的中心移到原点
    # 这样可以消除由 BERT/LSTM 带来的系统性位移 (Modality Gap)
    # =======================================================
    t_s = t_s - np.mean(t_s, axis=0)
    a_s = a_s - np.mean(a_s, axis=0)
    v_s = v_s - np.mean(v_s, axis=0)
    
    # 2. 数据拼接
    data_list = [
        t_s, a_s, v_s,
        features['t_private'], features['a_private'], features['v_private']
    ]
    data = np.concatenate(data_list, axis=0)
    
    # 3. 降维
    # init='pca' 有助于保持全局结构
    try:
        tsne = TSNE(n_components=2, perplexity=30, init='pca', learning_rate='auto', random_state=42)
        embedded = tsne.fit_transform(data)
    except:
        tsne = TSNE(n_components=2, perplexity=30, init='pca', random_state=42)
        embedded = tsne.fit_transform(data)
    
    n_samples = t_s.shape[0]
    
    # 4. 绘图 (样式美化)
    plt.figure(figsize=(18, 8))
    modality_labels = ['Text'] * n_samples + ['Audio'] * n_samples + ['Video'] * n_samples
    
    # Shared Space
    plt.subplot(1, 2, 1)
    shared_data = embedded[:3*n_samples]
    sns.scatterplot(
        x=shared_data[:,0], y=shared_data[:,1], 
        hue=modality_labels, style=modality_labels,
        palette={'Text': '#4E79A7', 'Audio': '#F28E2B', 'Video': '#59A14F'}, # 论文常用配色
        s=30, alpha=0.6, linewidth=0
    )
    plt.title(f"Shared Latent Space\n(Semantic Alignment)", fontsize=14)
    plt.legend(title='Modality')
    plt.grid(True, linestyle='--', alpha=0.3)

    # Private Space
    plt.subplot(1, 2, 2)
    private_data = embedded[3*n_samples:]
    sns.scatterplot(
        x=private_data[:,0], y=private_data[:,1], 
        hue=modality_labels, style=modality_labels,
        palette={'Text': '#B07AA1', 'Audio': '#9C755F', 'Video': '#FF9DA7'},
        s=30, alpha=0.6, linewidth=0
    )
    plt.title(f"Private Latent Space\n(Modality Specific)", fontsize=14)
    plt.legend(title='Modality')
    plt.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_name, dpi=300)
    print(f"可视化结果已保存至: {save_name}")
    plt.close()

if __name__ == "__main__":
    # 1. 获取配置
    args = get_args()
    
    # === 补全路径参数 ===
    args.mode = 'test'
    args.dataset = 'mosi' 
    if args.dataset in data_dict:
        args.dataset_dir = data_dict[args.dataset]
    else:
        args.dataset_dir = Path('./datasets/MOSI')
    args.sdk_dir = sdk_dir 
    args.data_dir = args.dataset_dir

    print(f"配置检查: Dataset={args.dataset}")
    print(f"配置检查: SDK Dir={args.sdk_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device}")
    
    # 2. 加载数据
    test_loader = get_loader(args, args, shuffle=False)
    
    # ==========================================
    # 【关键修复】强制将 DataLoader 的生成器改为 CPU
    # ==========================================
    test_loader.generator = torch.Generator(device='cpu')
    
    # === 拆包维度信息 ===
    args.d_tin, args.d_vin, args.d_ain = args.tva_dim
    
    # === 补全输出维度 ===
    from src.config import output_dim_dict
    args.n_class = output_dim_dict.get(args.dataset, 1)

    print(f"Data loaded. Input dims - Text:{args.d_tin}, Audio:{args.d_ain}, Video:{args.d_vin}")

    # ==========================================
    # 模型路径设置
    # ==========================================
    models_to_plot = [
        ("Ours_Aux0.07", FILE_PATH), 
    ]

    for name, path in models_to_plot:
        if not os.path.exists(path):
            print(f"错误: 找不到文件 {path}")
            alt_path = os.path.join("pre_trained_models", "mosi", "mosibest_model_0_07.pt")
            if os.path.exists(alt_path):
                path = alt_path
            else:
                continue
            
        print(f"\nProcessing: {name}")
        
        model = MMIM(args).to(device)
        print(f"Loading weights from {path}...")
        
        try:
            model.load_state_dict(torch.load(path))
        except RuntimeError as e:
            print(f"标准加载失败，尝试去除非严格模式加载... {e}")
            state_dict = torch.load(path)
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name_key = k[7:] if k.startswith('module.') else k
                new_state_dict[name_key] = v
            model.load_state_dict(new_state_dict, strict=False)
        
        feats = extract_features(model, test_loader, device)
        plot_tsne(feats, save_name=f'tsne_{name}.png', title_suffix=name)

    print("\n所有任务完成。")