import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.checkpoint import checkpoint
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
import os
import re
import numpy as np
from sklearn.model_selection import train_test_split
import torchvision.transforms as T
from torchvision.transforms import RandomErasing
import copy
import torch.nn.functional as F
from torchvision.transforms import ToPILImage

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import RandomErasing
import numpy as np
from PIL import Image
import os
import copy
import matplotlib.pyplot as plt
import pandas as pd
import json
import csv
import time
import torchvision.utils as vutils
import matplotlib.pyplot as plt
# import seaborn as sns
import cv2
from data_set2 import GeneDataset 

   

# -------------------- 配置参数 --------------------
config = {
    "batch_size": 4,
    "patch_size": (512, 512),
    "lr": 0.001,
    "lambda_consistency": 0.1,
    "num_epochs": 500,
    "num_workers": 4,
    "visualize_dir": '6.22data01',
    "topk": 500,
    "print_num":20,
    "out_put_dir": '7.5output_all'
}

# 定义提取函数
def extract_top_genes_batch(feature_maps, index_grid, name_grid, topk=10):
    """
    基于一批样本的中间特征图提取全局最重要的基因（修复版）
    """
    # 1. 使用ReLU确保非负激活
    non_negative_features = F.relu(feature_maps)
    
    # 2. 通道平均并插值
    heats = non_negative_features.mean(dim=1, keepdim=True)  # [B, 1, H, W]
    heat_resized = F.interpolate(
        heats, 
        size=index_grid.shape, 
        mode='bilinear', 
        align_corners=False
    )
    
    # 3. 计算全局平均热图（确保非负）
    global_heat = torch.mean(heat_resized, dim=0)[0]  # [H, W]
    global_heat = F.relu(global_heat).cpu().numpy()  # 确保非负
    
    # 4. 创建有效位置掩码
    valid_mask = np.ones_like(global_heat, dtype=bool)
    for i in range(global_heat.shape[0]):
        for j in range(global_heat.shape[1]):
            gene_idx = index_grid[i, j]
            gene_name = name_grid[i, j]
            # 标记无效位置：索引<2 或 名称为空
            if gene_idx < 2 or gene_name in ['', 'nan', None, np.nan]:
                valid_mask[i, j] = False
                global_heat[i, j] = -np.inf  # 设为最小值
    
    # 5. 展平并获取top基因（跳过无效位置）
    flat = global_heat.flatten()
    valid_indices = np.where(flat > -np.inf)[0]  # 只考虑有效位置
    
    # 如果有效基因不足topk，调整数量
    actual_topk = min(topk, len(valid_indices))
    top_idxs = flat[valid_indices].argsort()[::-1][:actual_topk]
    top_idxs = valid_indices[top_idxs]  # 映射回原始索引
    
    results = []
    for idx in top_idxs:
        i, j = divmod(int(idx), global_heat.shape[1])
        gene_idx = int(index_grid[i, j])
        gene_name = name_grid[i, j]
        
        # 安全处理基因名称
        if gene_name in ['', 'nan', None, np.nan]:
            gene_name = "Unknown"
        else:
            gene_name = str(gene_name)
        
        score = float(global_heat[i, j])
        results.append((gene_idx, gene_name, score))  # 注意：不再+2
    
    return results, global_heat
def mask_top_genes_in_image(image, gene_indices, index_grid, name_grid, img_size=(512,512)):
    """
    将输入图像中top基因对应的区域置零
    
    Args:
        image: 输入图像张量 [1, H, W]
        gene_indices: 要移除的基因索引列表
        index_grid: 基因网格索引矩阵
        name_grid: 基因名称网格
        img_size: 图像大小 (H, W)
    
    Returns:
        掩码后的图像
    """
    # 创建图像的副本
    masked_image = image.clone()
    H, W = img_size
    grid_h, grid_w = index_grid.shape
    
    # 计算每个基因块的大小
    block_h = H // grid_h
    block_w = W // grid_w
    
    # 遍历基因网格
    for i in range(grid_h):
        for j in range(grid_w):
            gene_idx = int(index_grid[i, j])
            # 如果当前基因在要移除的列表中
            if gene_idx + 2 in gene_indices:  # +2 对应csv文件中的索引
                # 计算该基因在图像中的位置
                h_start = i * block_h
                h_end = (i + 1) * block_h
                w_start = j * block_w
                w_end = (j + 1) * block_w
                
                # 将对应区域置零
                masked_image[:, h_start:h_end, w_start:w_end] = -0.1
                
    return masked_image

class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 16, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 16, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        att = self.attention(x)
        return x * att  # 特征加权


class ResidualBlock(nn.Module):
    def __init__(self, channels, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=dropout)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)


class GeneCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3),
            AttentionBlock(32),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            ResidualBlock(32, dropout=0.3),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            AttentionBlock(64),  # 中间层保留注意力
            ResidualBlock(64, dropout=0.3),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            ResidualBlock(128, dropout=0.3),
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)  # GAP
        self.classifier = nn.Linear(128, 2)  # 输出层

    # def forward(self, x):
    #     features = self.features(x)                 # [B, 128, H, W]
    #     pooled = self.global_pool(features)         # [B, 128, 1, 1]
    #     out = pooled.view(pooled.size(0), -1)       # [B, 128]
    #     return self.classifier(out), features       # 输出类别和特征
    
    #要是把第一次注意力后的图作为feature-map，forward这样改：
    def forward(self, x):
        attn_features = self.features[0:2](x)     # 到 AttentionBlock(32) 的输出
        features = self.features[2:](attn_features)  # 剩下的继续 forward
        pooled = self.global_pool(features)         # [B, 128, 1, 1]
        out = pooled.view(pooled.size(0), -1)       # [B, 128]
        return self.classifier(out), features, attn_features       # 输出类别和特征

# 可视化全局基因热图
def visualize_global_heatmap(global_heat, index_grid, name_grid, top_genes, save_path):
    plt.figure(figsize=(15, 10))
    
    # 绘制热图
    plt.imshow(global_heat, cmap='viridis')
    plt.colorbar(label='Activation Score')
    plt.title('Global Gene Activation Heatmap')
    
    # 标记top基因
    for gene_idx, gene_name, score in top_genes[:10]:
        # 查找基因在网格中的位置
        found = False
        for i in range(index_grid.shape[0]):
            for j in range(index_grid.shape[1]):
                if int(index_grid[i, j]) == gene_idx:  # 注意不再需要+2
                    # 检查是否有效基因
                    if gene_name != "Unknown" and not pd.isna(gene_name):
                        plt.scatter(j, i, s=100, c='red', edgecolors='white')
                        plt.text(j, i, gene_name, 
                                    fontsize=9, ha='center', va='bottom', color='white')
                        found = True
                    break
            if found:
                break
                    
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()





def load_model_and_mapping(model_path, device):
    """加载模型和基因映射网格"""
    student = GeneCNN().to(device)
    student.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    student.eval()
    
    mapping_dir = 'mix/mappings'
    index_grid = np.load(os.path.join(mapping_dir, 'index_grid.npy'))
    name_grid = np.load(os.path.join(mapping_dir, 'name_grid.npy'), allow_pickle=True)
    
    return student, index_grid, name_grid

def process_samples(student, dataloader, device, visualize_count):
    """处理样本并收集结果"""
    all_attn_features = []
    images_to_plot = []
    labels_to_plot = []
    preds_to_plot = []
    attn_maps = []
    sample_paths = []
    all_pred_scores = []  # 保存所有样本的预测分数
    total = 0
    
    with torch.no_grad():
        for idx, (images, labels, paths) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.to(device)
            
            # 获取模型输出
            outputs, features, attn_features = student(images)
            
            # 计算softmax概率（类别1的概率）
            probabilities = F.softmax(outputs, dim=1)
            class1_probs = probabilities[:, 1].cpu().numpy()
            
            # 保存预测分数
            all_pred_scores.extend(class1_probs)
            sample_paths.extend(paths)
            total += labels.size(0)
            
            # 保存用于全局分析的注意力特征（只保存前100个）
            if idx < 100:
                all_attn_features.append(attn_features.detach().cpu())
            
            # 保存用于可视化的样本（只保存部分样本）
            if len(images_to_plot) < visualize_count:
                # 提取注意力权重图
                attn = torch.mean(features, dim=1, keepdim=True)
                attn = F.interpolate(attn, size=config["patch_size"], mode="bilinear")
                attn = attn.squeeze(1).cpu().numpy()
                
                images_to_plot.extend(images.cpu().numpy())
                labels_to_plot.extend(labels.cpu().numpy())
                preds_to_plot.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                attn_maps.extend(attn)
            
            # 及时释放内存
            del outputs, features, attn_features
            torch.cuda.empty_cache()
    
    return {
        "all_attn_features": all_attn_features,
        "images_to_plot": images_to_plot,
        "labels_to_plot": labels_to_plot,
        "preds_to_plot": preds_to_plot,
        "attn_maps": attn_maps,
        "all_pred_scores": all_pred_scores,
        "sample_paths": sample_paths,
        "total_samples": total
    }

def save_prediction_scores(output_dir, results):
    """保存所有样本的预测分数到CSV文件"""
    if not results["all_pred_scores"]:
        print("⚠️ 没有预测分数可保存")
        return
    
    csv_path = os.path.join(output_dir, "sample_scores.csv")
    
    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["样本路径", "预测分数"])
        
        for i in range(results["total_samples"]):
            path = results["sample_paths"][i] if i < len(results["sample_paths"]) else f"sample_{i+1}"
            score = results["all_pred_scores"][i]
            writer.writerow([path, f"{score:.6f}"])
    
    print(f"📊 预测分数已保存到: {csv_path}")
    return csv_path

def analyze_global_genes(output_dir, all_attn_features, index_grid, name_grid):
    """分析并保存全局重要基因"""
    if not all_attn_features:
        print("⚠️ 没有足够的注意力特征进行全局基因分析")
        return
    
    # 合并所有注意力特征
    try:
        all_attn_features = torch.cat(all_attn_features, dim=0)
    except RuntimeError:
        print("⚠️ 无法拼接注意力特征")
        return
    
    # 提取全局重要基因
    top_genes, global_heat = extract_top_genes_batch(
        all_attn_features, index_grid, name_grid, topk=config['topk']
    )
    
    # 保存结果
    topk_file = os.path.join(output_dir, "top_genes.txt")
    with open(topk_file, "w", encoding="utf-8") as f:
        f.write(f"Top {config['topk']} 重要基因:\n")
        for rank, (gene_idx, gene_name, score) in enumerate(top_genes, start=1):
            f.write(f"{rank:03d}: GeneIdx={gene_idx}, GeneName={gene_name}, Score={score:.6f}\n")
    
    print(f"🧬 全局基因分析结果已保存到: {topk_file}")
    return topk_file

def visualize_sample_attention(output_dir, images_to_plot, attn_maps, labels_to_plot, preds_to_plot):
    """可视化样本的注意力图"""
    if not images_to_plot:
        print("⚠️ 没有样本可可视化")
        return 0
    
    num_visualized = min(len(images_to_plot), len(attn_maps), len(preds_to_plot), len(labels_to_plot))
    
    for i in range(num_visualized):
        img = images_to_plot[i][0]  # [1, H, W] -> [H, W]
        attn = attn_maps[i]
        pred = preds_to_plot[i]
        label = labels_to_plot[i]
        
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        
        # 原始图像
        ax[0].imshow(img, cmap='gray')
        ax[0].set_title(f"Image | Label: {label} | Pred: {pred}")
        
        # 注意力热图
        ax[1].imshow(img, cmap='gray')
        im = ax[1].imshow(attn, cmap='jet', alpha=0.5)
        ax[1].set_title("Attention Overlay")
        
        # 添加颜色条
        fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)
        
        for a in ax:
            a.axis('off')
        plt.tight_layout()
        
        # 保存图像
        filename = os.path.join(output_dir, f"sample_{i+1}_label{label}_pred{pred}.png")
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    print(f"🖼️ 已可视化 {num_visualized} 个样本的注意力图")
    return num_visualized

def test_and_visualize(model_path, dataloader, device, visualize_count=config['print_num']):
    """主测试与可视化函数"""
    # 1. 准备输出目录
    output_dir = config["out_put_dir"]
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 输出目录: {output_dir}")
    
    # 2. 加载模型和基因映射
    print("🔍 加载模型和基因映射...")
    student, index_grid, name_grid = load_model_and_mapping(model_path, device)
    
    # 3. 处理样本数据
    print("🔍 处理样本数据...")
    results = process_samples(student, dataloader, device, visualize_count)
    
    # 4. 保存预测分数
    print("💾 保存预测分数...")
    scores_file = save_prediction_scores(output_dir, results)
    
    # 5. 分析全局重要基因
    print("🔬 分析全局重要基因...")
    # genes_file = analyze_global_genes(output_dir, results["all_attn_features"], index_grid, name_grid)
    
    # 6. 可视化样本注意力图
    print("🎨 可视化样本注意力图...")
    # num_visualized = visualize_sample_attention(
    #     output_dir,
    #     results["images_to_plot"],
    #     results["attn_maps"],
    #     results["labels_to_plot"],
    #     results["preds_to_plot"]
    # )
    
    # 7. 完成报告
    print("\n✅ 测试完成! 结果:")
    print(f"  - 样本总数: {results['total_samples']}")
    print(f"  - 预测分数文件: {scores_file}")
    # if genes_file:
    #     print(f"  - 基因分析文件: {genes_file}")
    # print(f"  - 可视化样本数: {num_visualized}")



if __name__ == "__main__":
    os.makedirs(config["out_put_dir"], exist_ok=True)
    # 加载验证集或测试集（和训练时一致）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dataset = GeneDataset( img_dir="6.16with_label",
                                mode='test',
                                transform=T.ToTensor(),
                                use_all_data= True,  # 使用全部数据
                                )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    print("📦 测试集样本数量：", len(test_dataset))
   
    # 调用测试 + 可视化函数
    test_and_visualize("/home/wangshaoxiang_t/wsx-porject/1/6.22data01/best_model_epoch35_acc0.73.pth", test_loader, device)

