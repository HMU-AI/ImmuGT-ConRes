'''
Author: ZhangLan,WangShaoXiang
Date: 2025-04-11 07:52:27
LastEditors: WangShaoXiang
LastEditTime: Thu 24 Apr 2025 04:59:25 PM CST
Description: 将csv件中的每一个病患基因保存成图像，图片的命名按照患者的ID命名，其中N开头的是正常基因，C开头的是癌症基因
             两段代码命名规则不同，第一个是按照序号命名，第二个是按照患者ID命名
             像素的处理：标准化，扩大动态范围，分位数裁剪
'''

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


input_dir = 'data'
output_dir = '5.30data_picture'


# 0. 加载 “主基因列表” 
MASTER_GENE_CSV = 'res.csv'  
# 假设第一列包含所有可能用到的基因名
master_df = pd.read_csv(MASTER_GENE_CSV, usecols=[0], header=None, names=['gene'])
# 统一大小写，消除大小写不一致的问题
master_df['gene_upper'] = master_df['gene'].str.upper()
master_genes = master_df['gene_upper'].tolist()

root_data_dir = input_dir
subfolders = [f for f in os.listdir(root_data_dir) if os.path.isdir(os.path.join(root_data_dir, f))]

for subfolder in subfolders:
    INPUT_CSV = os.path.join(root_data_dir, subfolder, 'expression.csv')
    if not os.path.exists(INPUT_CSV):
        print(f"⚠️ 文件不存在，跳过: {INPUT_CSV}")
        continue

    # INPUT_CSV = 'biaodapu/TCGA_STAD_expr_data.csv'
    os.makedirs(output_dir, exist_ok=True)
    #在开头解析 INPUT_CSV 文件名的前缀
    csv_basename = subfolder  # 修改：使用子文件夹名作为前缀


    # 1.读取数据并转置与标准化
    df = pd.read_csv(INPUT_CSV, index_col=0,low_memory=False)  #  GSE15459_expr_data,  GSE26253_expr_data  ,   GSE26899_expr_data,,   GSE26901_expr_data  ,   GSE28541_expr_data  ,  GSE29272_expr_data   ,  GSE34942_expr_data   ,     GSE38749_expr_data   , GSE57303_expr_data  ,  GSE62254_expr_data   ,  GSE84437_expr_data  ,   GSE183136_expr_data    ,   TCGA_STAD_expr_data
    df = df.T  # 转置后，行=样本，列=基因

    df.columns = df.columns.astype(str).str.upper()

    # 重新索引到主列表，用 0 填充缺失的基因
    # df = df.reindex(columns=master_genes, fill_value=0)

    # 标准化
    scaler = MinMaxScaler(feature_range=(-2, 2))
    normalized_data = scaler.fit_transform(df)
    normalized_df = pd.DataFrame(normalized_data, index=df.index, columns=df.columns)

    print("数据尺寸:", df.shape)
    print("样本名示例:", df.index[:5])  # 检查样本名是否正确

    # 2. 构建统一维度的映射矩阵并保存像素↔基因映射矩阵
    genes_per_row = 100
    n_genes = len(master_genes)
    n_rows = int(np.ceil(n_genes / genes_per_row))

    # 索引映射 0…n_genes-1 对应 master_genes
    flat_idx = np.arange(n_genes)
    idx_padded = np.pad(flat_idx, (0, n_rows*genes_per_row - n_genes), constant_values=-1)
    index_grid = idx_padded.reshape(n_rows, genes_per_row)

    # 名称映射：使用主列表顺序
    name_padded = np.pad(
        master_genes,
        (0, n_rows*genes_per_row - n_genes),
        constant_values=''
    )
    name_grid = name_padded.reshape(n_rows, genes_per_row)

    mapping_dir = os.path.join(output_dir, 'mappings')
    os.makedirs(mapping_dir, exist_ok=True)
    index_grid_path = os.path.join(mapping_dir, 'index_grid.npy')
    name_grid_path = os.path.join(mapping_dir, 'name_grid.npy')
    # ✅ 只在文件不存在时保存一次
    if not os.path.exists(index_grid_path) or not os.path.exists(name_grid_path):
        np.save(index_grid_path, index_grid)
        np.save(name_grid_path, name_grid)
        print(f"✔ 映射矩阵已保存到：{mapping_dir}")
    else:
        print(f"✔ 映射矩阵已存在，跳过保存：{mapping_dir}")
    
    # 3.生成图像函数
    def genes_to_image(patient_data, genes_per_row=100):
        n_genes = len(patient_data)
        n_rows = int(np.ceil(n_genes / genes_per_row))
        padded = np.pad(patient_data, (0, n_rows * genes_per_row - n_genes))
        image_2d = padded.reshape(n_rows, genes_per_row)
        # 分位数裁剪
        low, high = np.quantile(image_2d, 0.1), np.quantile(image_2d, 0.9)
        return np.clip(image_2d, low, high)


    # 4.遍历每个样本（使用样本名而非数字索引）
    for sample_name in normalized_df.index:
        vec = normalized_df.loc[sample_name].values
        img = genes_to_image(vec, genes_per_row)

        plt.figure(figsize=(12, 12))
        plt.imshow(img, cmap='viridis')
        plt.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        safe_name = str(sample_name).replace('/', '_').replace(':', '_')
        out_path = os.path.join(output_dir, f"{csv_basename}_{safe_name}_heatmap.png")
        plt.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()

    print(f"✔ {subfolder}: 已保存 {len(normalized_df)} 张热图到目录: {output_dir}")




