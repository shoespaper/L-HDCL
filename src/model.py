import torch
from torch import nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from .modules.encoders import LanguageEmbeddingLayer, CPC, MMILB, RNNEncoder, SubNet, Clip, get_resNet, get_resNet2, get_resNet3, get_resNet4, DecouplingLayer, OrthogonalLoss

from transformers import BertModel, BertConfig
from .modules.transformer import TransformerEncoder
import torchvision
from torchvision import models


def align_loss(x, y, alpha=2):
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()


def uniform_loss(x, t=2):
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()



class MMIM(nn.Module):
    def __init__(self, hp):
        """Construct MultiMoldal InfoMax model (Modified L-HDCL Version)."""
        super().__init__()
        self.hp = hp
        
        # ================= 1. 基础编码器 (保留不变) =================
        # 文本编码器 (BERT)
        self.text_enc = LanguageEmbeddingLayer(hp) 
        self.hp.d_tout = self.hp.d_tin # 通常是 768

        # 视觉编码器 (LSTM)
        self.visual_enc = RNNEncoder(
            in_size=hp.d_vin,
            hidden_size=hp.d_vh,
            out_size=hp.d_vout, # 假设这里是 32 或 64，取决于你的config
            num_layers=hp.n_layer,
            dropout=hp.dropout_v if hp.n_layer > 1 else 0.0,
            bidirectional=hp.bidirectional
        )
        # 音频编码器 (LSTM)
        self.acoustic_enc = RNNEncoder(
            in_size=hp.d_ain,
            hidden_size=hp.d_ah,
            out_size=hp.d_aout, # 同上
            num_layers=hp.n_layer,
            dropout=hp.dropout_a if hp.n_layer > 1 else 0.0,
            bidirectional=hp.bidirectional
        )

        # ================= 2. 特征解耦模块 (新增核心) =================
        # 设定统一的公共特征维度，用于对比学习和融合
        self.common_dim = hp.common_dim
        
        # 定义解耦层
        # 注意：这里输入维度要和你配置文件(hp)里的输出维度对应
        self.t_decoupler = DecouplingLayer(in_dim=hp.d_tin, common_dim=self.common_dim)
        self.v_decoupler = DecouplingLayer(in_dim=hp.d_vout, common_dim=self.common_dim)
        self.a_decoupler = DecouplingLayer(in_dim=hp.d_aout, common_dim=self.common_dim)
        
        # 正交损失计算器
        self.orth_loss = OrthogonalLoss()

        # ================= 3. 第一级对比学习 (Level-1) =================
        # 输入都是 common_dim，所以 Clip 初始化很简单
        self.clip_ta = Clip(self.common_dim, self.common_dim)
        self.clip_tv = Clip(self.common_dim, self.common_dim)
        self.clip_av = Clip(self.common_dim, self.common_dim)

        # ================= 4. 第二级对比学习 (Level-2 全对称) =================
        # 我们需要将融合后的特征投影回 common_dim 才能进行 Clip 计算
        
        # 两两融合后的投影层 (输入是 2 * common_dim)
        self.proj_tv = nn.Linear(self.common_dim * 2, self.common_dim)
        self.proj_ta = nn.Linear(self.common_dim * 2, self.common_dim)
        self.proj_av = nn.Linear(self.common_dim * 2, self.common_dim)
        
        # 三模态浅层融合投影层 (输入是 3 * common_dim)
        self.proj_tav = nn.Linear(self.common_dim * 3, self.common_dim)

        # 定义 Level-2 的 Clip 模块 (对应你的全对称设计)
        # 融合特征 vs 单模态特征
        self.clip_l2_tv_t = Clip(self.common_dim, self.common_dim) # TV vs T
        self.clip_l2_tv_v = Clip(self.common_dim, self.common_dim) # TV vs V
        # ... 为了代码简洁，可以在 forward 里复用部分 Clip，但建议分开定义以避免参数耦合
        # 这里我们简化一下，只定义核心的 Clip，具体复用在 forward 展示
        self.clip_fusion_unimodal = Clip(self.common_dim, self.common_dim)

        # ================= 5. 第三级轻量级融合 (Level-3 & Transformer) =================
        # 替代 TCF 模块
        # 定义一个单层的 Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.common_dim, nhead=4, dim_feedforward=256, dropout=0.1)
        self.fusion_transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
        # Level-3 对比学习 (高级融合特征 vs 单模态共享特征)
        self.clip_l3 = Clip(self.common_dim, self.common_dim)

        # ================= 6. 最终预测层 =================
        # 输入是：高级融合特征(Shared) + 3个私有特征(Private)
        # 维度 = 128 + 128 + 128 + 128 = 512
        final_in_dim = self.common_dim * 4
        
        self.fusion_prj = nn.Sequential(
            nn.Linear(final_in_dim, 128),
            nn.ReLU(),
            nn.Dropout(hp.dropout_prj),
            nn.Linear(128, hp.n_class) # 输出情感得分
        )

        # ================= 7. 动态权重参数 (新增) =================
        # 我们有5个主要的 Loss 部分: Task, Diff, L1, L2, L3
        # 初始化为 0 (即权重为 1)
        # 忽略此警告
        self.loss_log_vars = nn.Parameter(torch.zeros(5))

        # ================= 8. 清理掉的代码 (原 TCF 部分) =================
        # 下面这些原有的代码全部删掉，节省显存和参数量
        # self.conv_tv, self.conv_av ... (删除)
        # self.getresNet ... (删除)
        # self.conv_tfn_t ... (删除)
        # self.tfn_tv ... (删除)
        # resnet50 ... (删除)

    def forward(self, sentences, visual, acoustic, v_len, a_len, bert_sent, bert_sent_type, bert_sent_mask, y=None, return_features=False):
        # 1. 基础特征提取
        # 文本
        enc_word = self.text_enc(sentences, bert_sent, bert_sent_type, bert_sent_mask)
        text_raw = enc_word[:, 0, :] # [B, 768]
        # 音频 & 视觉
        acoustic_raw, _ = self.acoustic_enc(acoustic, a_len) # [B, d_aout]
        visual_raw, _ = self.visual_enc(visual, v_len)       # [B, d_vout]

        # 2. 特征解耦
        t_shared, t_private = self.t_decoupler(text_raw)       # [B, 128]
        a_shared, a_private = self.a_decoupler(acoustic_raw)   # [B, 128]
        v_shared, v_private = self.v_decoupler(visual_raw)     # [B, 128]

        # === 【修改点】如果是为了画图，直接在这里返回特征，跳过后面计算 ===
        if return_features:
            return {
                't_shared': t_shared, 't_private': t_private,
                'a_shared': a_shared, 'a_private': a_private,
                'v_shared': v_shared, 'v_private': v_private
            }
        # ==========================================================

        # 计算差异性 Loss (Diff Loss)
        loss_diff = self.orth_loss(t_shared, t_private) + \
                    self.orth_loss(a_shared, a_private) + \
                    self.orth_loss(v_shared, v_private)

        # 3. Level-1 对比学习 (Unimodal)
        l1_ta = self.clip_ta(t_shared, a_shared)
        l1_tv = self.clip_tv(t_shared, v_shared)
        l1_av = self.clip_av(a_shared, v_shared)
        loss_l1 = l1_ta + l1_tv + l1_av

        # 4. Level-2 对比学习 (Symmetric Bimodal)
        # 拼接并投影
        h_tv = self.proj_tv(torch.cat([t_shared, v_shared], dim=1))
        h_ta = self.proj_ta(torch.cat([t_shared, a_shared], dim=1))
        h_av = self.proj_av(torch.cat([a_shared, v_shared], dim=1))
        h_tav = self.proj_tav(torch.cat([t_shared, a_shared, v_shared], dim=1))

        # 计算全对称 Loss
        # 这里复用 clip_fusion_unimodal 来计算相似度
        # 三模态融合 vs 单模态 (你的全对称改进点)
        l2_tav_t = self.clip_fusion_unimodal(h_tav, t_shared)
        l2_tav_a = self.clip_fusion_unimodal(h_tav, a_shared)
        l2_tav_v = self.clip_fusion_unimodal(h_tav, v_shared)
        
        # 也可以加上两模态的对比 (可选，为了简洁这里只写核心的三模态改进)
        loss_l2 = (l2_tav_t + l2_tav_a + l2_tav_v) / 3.0

        # 5. Level-3 轻量级 Transformer 融合
        # 构造序列: [Seq_len=3, Batch, Dim=128]
        seq_input = torch.stack([t_shared, a_shared, v_shared], dim=0) 
        
        # Transformer 编码
        # out: [3, B, 128]
        enc_out = self.fusion_transformer(seq_input)
        
        # 获取高级融合特征 (取平均，或者取第一个位置)
        h_final_shared = enc_out.mean(dim=0) # [B, 128]

        # Level-3 对比 Loss
        l3_t = self.clip_l3(h_final_shared, t_shared)
        l3_a = self.clip_l3(h_final_shared, a_shared)
        l3_v = self.clip_l3(h_final_shared, v_shared)
        loss_l3 = l3_t + l3_a + l3_v

        # 6. 最终预测
        # 拼接: 高级共享特征 + 3个私有特征
        final_vec = torch.cat([h_final_shared, t_private, a_private, v_private], dim=1) # [B, 512]
        preds = self.fusion_prj(final_vec)

        # 7. 返回结果
        # 注意：这里返回的是 各个 Loss 组件，方便在 train.py 里面用动态权重组合
        # 如果你的 train.py 是直接取 total loss，那我们在这里组合
        
        return preds, loss_diff, loss_l1, loss_l2, loss_l3, self.loss_log_vars
    
    
        # """
        # text, audio, and vision should have dimension [batch_size, seq_len, n_features]
        # For Bert input, the length of text is "seq_len + 2"
        # """  # (32,50,768)
        # enc_word = self.text_enc(sentences, bert_sent, bert_sent_type,
        #                          bert_sent_mask)  # (batch_size, seq_len, emb_size)->(32,50,768)
        
        # text = enc_word[:, 0, :]  # (batch_size, emb_size)32,768
        
        # acoustic, aco_rnn_output = self.acoustic_enc(
        #     acoustic, a_len)  # (218,32,5)-> (32,64) #aco_rnn_output->[32, 64, 218])
        # visual, vis_rnn_output = self.visual_enc(
        #     visual, v_len)  # (261,32,20)-> (32,64) vis_rnn_output->[261, 32, 16])  49152

        # massagehub_tv = torch.cat([text, visual], dim=1)  # 32,832
        # massagehub_va = torch.cat([visual, acoustic], dim=1)  # 32,128
        # massagehub_ta = torch.cat([text, acoustic], dim=1)  # 32,832
        # massagehub_tav = torch.cat([text, visual, acoustic], dim=1)
        # massagehub_tv = self.conv_tv(
        #     massagehub_tv.unsqueeze(2)).squeeze(2)  # 32,64
        # massagehub_va = self.conv_av(massagehub_va.unsqueeze(2)).squeeze(2)
        # massagehub_ta = self.conv_vt(massagehub_ta.unsqueeze(2)).squeeze(2)
        # massagehub_tav = self.conv_tav(massagehub_tav.unsqueeze(2)).squeeze(2)

        # # fusion, preds = self.fusion_prj(
        # #     torch.cat([text, acoustic, visual], dim=1))  # 32,896)
        # tav = torch.cat([text, acoustic, visual], dim=1).unsqueeze(2)  # 32,896
        # qwer = self.getresNet4(tav).squeeze(2)

        # res = self.getresNet(text.unsqueeze(2)).squeeze(2)  # 32 512
        # res2 = self.getresNet2(visual.unsqueeze(2)).squeeze(2)  # 32 512
        # res3 = self.getresNet3(acoustic.unsqueeze(2)).squeeze(2)  # 32 512

        # tfn_text_visual = torch.bmm(
        #     res.unsqueeze(2), res2.unsqueeze(1)).unsqueeze(1)  # [32, 1,512, 512]
        # tfn_text_acoustic = torch.bmm(
        #     res.unsqueeze(2), res3.unsqueeze(1)).unsqueeze(1)  # [32,1, 512, 512]
        # tfn_visual_acoustic = torch.bmm(
        #     res2.unsqueeze(2), res3.unsqueeze(1)).unsqueeze(1)  # [32, 1,512, 512]
        # tfn_text_visual = self.conv_tfn_t(
        #     tfn_text_visual)  # [32, 16386304]  [32, 2048288] 32, 2000000 32, 1952288 1016064
        # tfn_text_acoustic = self.conv_tfn_v(
        #     tfn_text_acoustic)
        # tfn_visual_acoustic = self.conv_tfn_a(
        #     tfn_visual_acoustic)

        # res_all = torch.cat([res, res2, res3], dim=1)  # [32, 1536]
        # xxx = torch.cat([res_all, qwer], dim=1).unsqueeze(2)
        # res_mm = self.conv_res(xxx).squeeze(2)

        # fusion, preds = self.fusion_prj(res_mm)  # 32, 512, 1

        # clip_ta = self.ta_clip(text, acoustic)
        # clip_tv = self.tv_clip(text, visual)
        # clip_av = self.av_clip(visual, acoustic)

        # clip_mass_t = self.mass_t_clip(text, massagehub_va)
        # clip_mass_v = self.mass_v_clip(visual, massagehub_ta)
        # clip_mass_a = self.mass_a_clip(acoustic, massagehub_tv)

        # clip_mass_tav = self.mass_tav_clip(acoustic, massagehub_tav)

        # tfn_tv_loss = self.tfn_tv(tfn_text_visual, tfn_text_acoustic)
        # tfn_ta_loss = self.tfn_ta(tfn_text_visual, tfn_visual_acoustic)
        # tfn_av_loss = self.tfn_av(tfn_text_acoustic, tfn_visual_acoustic)

        # # tfn_tav_loss = self.tfn_tav(
        # #     tfn_text_visual_acoustic, tfn_text_visual_acoustic)

        # nce = clip_ta+clip_tv + clip_av
        # nce2 = clip_mass_t+clip_mass_v+clip_mass_a+clip_mass_tav

        # nce3 = tfn_tv_loss+tfn_ta_loss+tfn_av_loss

        # return nce, preds, nce2, nce3
