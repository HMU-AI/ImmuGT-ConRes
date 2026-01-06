from torch.utils.data import Dataset
from torch.utils.checkpoint import checkpoint
from PIL import Image
import os
import re
import csv
from sklearn.model_selection import train_test_split
import pickle

# -------------------- 配置参数 --------------------
config = {
    "batch_size": 4,
    "patch_size": (512, 512),
    "lr": 0.001,
    "lambda_consistency": 0.1,
    "num_epochs": 500,
    "num_workers": 4
}

# -------------------- 自动划分训练集/验证集 --------------------
class GeneDataset(Dataset):
    def __init__(self, img_dir, transform=None, mode='train', train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, 
                 random_state=42, split_cache_path='splits.pkl', use_all_data=False):
        
        self.image_dir = img_dir
        self.transform = transform
        self.mode = mode
        
        # 获取所有PNG文件路径
        image_paths = [f for f in os.listdir(self.image_dir) if f.endswith('.png')]
        
        if use_all_data:
            self.image_paths = image_paths
            self.labels = self._generate_labels(image_paths)
            return
        
        if os.path.exists(split_cache_path):
            with open(split_cache_path, 'rb') as f:
                splits = pickle.load(f)
            self.image_paths = splits[mode]['paths']
            self.labels = splits[mode]['labels']
        else:
            # 首先生成所有标签
            all_labels = self._generate_labels(image_paths)
            
            # 只包含有效标签的文件
            valid_indices = [i for i, label in enumerate(all_labels) if label is not None]
            valid_paths = [image_paths[i] for i in valid_indices]
            valid_labels = [all_labels[i] for i in valid_indices]
            
            # train/test+val split
            train_paths, temp_paths, train_labels, temp_labels = train_test_split(
                valid_paths, valid_labels,
                test_size=(1.0 - train_ratio),
                stratify=valid_labels,
                random_state=random_state
            )

            # val/test split
            val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
            val_paths, test_paths, val_labels, test_labels = train_test_split(
                temp_paths, temp_labels,
                test_size=(1.0 - val_ratio_adjusted),
                stratify=temp_labels,
                random_state=random_state
            )

            splits = {
                'train': {'paths': train_paths, 'labels': train_labels},
                'val': {'paths': val_paths, 'labels': val_labels},
                'test': {'paths': test_paths, 'labels': test_labels}
            }

            with open(split_cache_path, 'wb') as f:
                pickle.dump(splits, f)
                
            self.image_paths = splits[mode]['paths']
            self.labels = splits[mode]['labels']

    def _generate_labels(self, image_paths):
        """直接从文件名前缀生成标签"""
        labels = []
        for img_path in image_paths:
            # 检查文件名前缀
            if img_path.startswith('0_'):
                labels.append(0)  # YES
            elif img_path.startswith('1_'):
                labels.append(1)  # NO
            else:
                # 没有有效前缀的文件
                print(f"警告: 文件 {img_path} 没有有效标签前缀，将被排除")
                labels.append(None)  # 标记为无效
                
        return labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_paths[idx])
        image = Image.open(img_path).convert('L')
        
        # 中心裁剪
        w, h = image.size
        left = (w - config["patch_size"][1]) // 2
        top = (h - config["patch_size"][0]) // 2
        image = image.crop((left, top, left+config["patch_size"][1], top+config["patch_size"][0]))
        
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx], self.image_paths[idx]