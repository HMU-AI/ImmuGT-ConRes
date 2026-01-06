import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.checkpoint import checkpoint
from sklearn.model_selection import train_test_split
import torchvision.transforms as T
from torchvision.transforms import RandomErasing
import copy
import torch.nn.functional as F
from torchvision.transforms import ToPILImage
from sklearn.model_selection import train_test_split
import json
import csv
import time
from data_set2 import GeneDataset 

# -------------------- 配置参数 --------------------
config = {
    "batch_size": 4,
    "patch_size": (512, 512),
    "lr": 0.001,
    "lambda_consistency": 0.1,
    "num_epochs": 500,
    "num_workers": 4
}


# -------------------- 基因数据增强（返回弱增强+强增强+标签） --------------------
class GeneAugment:
    def __init__(self):
        self.to_pil = ToPILImage()
        # 弱增强（轻微变换）
        self.weak = T.Compose([
            T.RandomHorizontalFlip(p=0.3),
            T.RandomVerticalFlip(p=0.3),
            T.RandomAffine(degrees=0, translate=(0.05, 0.05)),  # 小幅平移
            T.ToTensor(),
        ])
        # 强增强（遮挡+噪声）
        self.strong = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 更大平移
            
            T.ToTensor(),
            T.Lambda(lambda x: x + torch.randn_like(x) * 0.03),# 高斯噪声
            RandomErasing(p=0.5, scale=(0.02, 0.15))  # 遮挡
            
        ])

    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            img = self.to_pil(img)
        return self.weak(img), self.strong(img)

# -------------------- 注意力模块 --------------------
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

# #---------------------- 残差结构--------------------------
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

# # -------------------- 残差主干网络（CNN + 注意力）--------------------
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

    def forward(self, x):
        features = self.features(x)                 # [B, 128, H, W]
        pooled = self.global_pool(features)         # [B, 128, 1, 1]
        out = pooled.view(pooled.size(0), -1)       # [B, 128]
        return self.classifier(out), features       # 输出类别和特征



 
def focal_loss(inputs, targets, alpha=1, gamma=2, reduction='mean'):
    """
    inputs: raw logits (no softmax), shape [batch_size, num_classes]
    targets: ground truth labels, shape [batch_size]
    alpha: balance factor
    gamma: focusing parameter
    """
    ce_loss = F.cross_entropy(inputs, targets, reduction='none')  # [B]
    pt = torch.exp(-ce_loss)  # pt = softmax probability of the correct class
    focal = alpha * (1 - pt) ** gamma * ce_loss

    if reduction == 'mean':
        return focal.mean()
    elif reduction == 'sum':
        return focal.sum()
    else:
        return focal  # no reduction    

# -------------------- Mean Teacher 模型 --------------------
class MeanTeacher(nn.Module):
    def __init__(self, student):
        super().__init__()
        self.student = student
        self.teacher = copy.deepcopy(student)
        self.alpha = 0.95  # EMA衰减系数
        
        # 冻结教师模型梯度
        for param in self.teacher.parameters():
            param.requires_grad = False

    def update_teacher(self):
        # EMA更新教师模型
        with torch.no_grad():
            for t_param, s_param in zip(self.teacher.parameters(), self.student.parameters()):
                t_param.data = self.alpha * t_param.data + (1 - self.alpha) * s_param.data

    def forward(self, x_weak, x_strong, labels=None):
        # 教师模型（弱增强）
        with torch.no_grad():
            _, feat_teacher = self.teacher(x_weak)
        
        # 学生模型（强增强）
        preds, feat_student = self.student(x_strong)
        # preds = self.student(x_strong)  # 学生模型分类预测
        if labels is None:
            print("labels is None")
        # print("labels",labels)
        # print("preds",preds)
        # 分类损失（CrossEntropy）
        # loss_cls = F.cross_entropy(preds, labels) if labels is not None else 0
        loss_cls = focal_loss(preds, labels) if labels is not None else 0
        
        # 特征一致性损失（MSE）
        loss_consistency = F.mse_loss(feat_student, feat_teacher)
        
        return preds, loss_cls, loss_consistency
        
# -------------------- 训练函数 --------------------
def train(model, train_loader, optimizer, device, epoch, csv_writer):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    criterion = nn.CrossEntropyLoss()

    for batch_idx, (images, labels,__paths) in enumerate(train_loader):
        # 将图像和标签转移到设备
        # images, labels = images.to(device), labels.to(device)
        
        # 生成增强样本
        augment = GeneAugment()
        x_weak = torch.empty(0)#.to(device)
        x_strong = torch.empty(0)#.to(device)
        
        for image in images:
            w, s = augment(image)
            x_weak = torch.cat((x_weak, w.unsqueeze(0)), dim=0)
            x_strong = torch.cat((x_strong, s.unsqueeze(0)), dim=0)
        x_weak, x_strong, labels = x_weak.to(device), x_strong.to(device), labels.to(device)
        images = images.to(device)
        # 前向传播
        optimizer.zero_grad()
        preds, loss_cls, loss_consistency = model(images, x_strong, labels)
        
        # 总损失 = 分类损失 + 0.1 * 一致性损失
        loss = loss_cls #+ 0.1 * loss_consistency
        # if epoch < 20:
        #     loss = loss_cls  # 前20个 epoch 只用分类损失
        # else:
        #     # print("a")
        #     loss = loss_cls + 0.1 * loss_consistency
        # 反向传播
        loss.backward()
        optimizer.step()
        model.update_teacher()  # 更新教师模型

        # 统计指标
        total_loss += loss.item()
        total_acc += (preds.argmax(dim=1) == labels).float().mean().item()
        # if csv_writer:
        # csv_writer.writerow([epoch+1, batch_idx+1, loss.item(), loss_cls.item(), loss_consistency.item()])

        # 打印当前的损失和准确率
        if batch_idx % 10 == 0:  # 每10个batch打印一次
            print(f"Epoch {epoch+1}, Batch {batch_idx+1}/{len(train_loader)}:")
            print(f"    Loss: {loss.item():.4f}, Loss_cls: {loss_cls.item():.4f}, Loss_consistency: {loss_consistency.item():.4f}")
    
    return total_loss / len(train_loader), total_acc / len(train_loader)

# -------------------- 验证函数 --------------------
def validate(model, train_loader, device ):#epoch, csv_writer
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    # criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch_idx, (images, labels,__paths) in enumerate(train_loader):
        # 将图像和标签转移到设备
        # images, labels = images.to(device), labels.to(device)
        
        # 生成增强样本
            augment = GeneAugment()
            x_weak = torch.empty(0)#.to(device)
            x_strong = torch.empty(0)#.to(device)
            
            for image in images:
                w, s = augment(image)
                x_weak = torch.cat((x_weak, w.unsqueeze(0)), dim=0)
                x_strong = torch.cat((x_strong, s.unsqueeze(0)), dim=0)
            x_weak, x_strong, labels = x_weak.to(device), x_strong.to(device), labels.to(device)
            images = images.to(device)
            # 前向传播
            # optimizer.zero_grad()
            preds, loss_cls , _ = model(images, x_strong, labels)
            total_loss += loss_cls.item()
            total_acc += (preds.argmax(dim=1) == labels).float().mean().item()
            # 总损失 = 分类损失 + 0.1 * 一致性损失
            loss = total_loss
            print("loss",loss_cls.item())
 
           
    return  total_acc / len(train_loader) , preds
 
# -------------------- 主函数 --------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_dir = "data_picture"  # 图像数据目录
    # 自动划分训练集/验证集
    train_dataset = GeneDataset(img_dir, mode='train', transform=T.ToTensor())  #gene_heatmaps
    val_dataset = GeneDataset(img_dir, mode='val', transform=T.ToTensor())
    test_dataset = GeneDataset(img_dir, mode='test', transform=T.ToTensor())

    from collections import Counter
    print("Train label distribution:", Counter(train_dataset.labels))
    print("Val label distribution:", Counter(val_dataset.labels))
    print("Test label distribution:", Counter(test_dataset.labels))

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # 初始化模型和优化器
    student = GeneCNN().to(device)
    model = MeanTeacher(student).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Early Stopping 设置
    best_acc = 0.0
    patience = 10
    patience_counter = 0

    with open("4-18.csv", mode='a', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(["Epoch", "Step", "Loss", "Loss_cls", "Loss_consistency"])

        for epoch in range(500):
            train_loss, train_acc = train(model, train_loader, optimizer, device, epoch, csv_writer)
            val_acc, _ = validate(model, val_loader, device)

            print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2%}, Val Acc={val_acc:.2%}")
            row = {
                "Epoch": epoch + 1,
                "Train Acc": train_acc,
                "Val Acc": val_acc
            }
            writer = csv.DictWriter(file, fieldnames=row.keys())
            writer.writerow(row)

            if val_acc > best_acc:
                best_acc = val_acc
                patience_counter = 0

                # 保存模型文件（包含 acc 和 epoch）
                model_path = f"best_model_epoch{epoch+1}_acc{val_acc:.2f}.pth"
                torch.save(model.student.state_dict(), model_path)
                print(f"✅ New best model saved: {model_path}")

                # 保存附加信息
                meta_info = {
                    "epoch": epoch + 1,
                    "val_acc": round(val_acc, 4),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                    "config": config
                }
                with open("best_model_info.json", "w") as json_file:
                    json.dump(meta_info, json_file, indent=4)

            else:
                patience_counter += 1
                print(f"⏳ No improvement for {patience_counter} epoch(s)")

            if patience_counter >= patience:
                print("⛔ Early stopping triggered.")
                break

    print(f"🏆 Best Validation Accuracy: {best_acc:.2%}")
