import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# ==========================================
# 模块 1：数据读取 (原封不动继承上一节)
# ==========================================
class CatDogDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_paths, self.labels = [], []
        self.transform = transform
        class_to_idx = {"cats": 0, "dogs": 1}
        
        for class_name, idx in class_to_idx.items():
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                for file_name in os.listdir(class_dir):
                    if file_name.endswith('.jpg'):
                        self.image_paths.append(os.path.join(class_dir, file_name))
                        self.labels.append(idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]

# ==========================================
# 模块 2：网络结构 (我们手撕的 ResNet 核心)
# ==========================================
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)

class MiniResNet(nn.Module):
    """将残差块包装成一个能直接输出分类结果的完整网络"""
    def __init__(self, num_classes=2):
        super().__init__()
        # 1. 进门第一层：把 3 通道图片砸成 64 通道特征图
        self.entry_conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.entry_bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # 2. 核心特征提取：接入我们的残差块
        self.res_block = ResidualBlock(in_channels=64, out_channels=64)
        
        # 3. 汇总与分类：把 128x128 的特征图浓缩，最后全连接层输出 2 个类别得分
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.relu(self.entry_bn(self.entry_conv(x))) # 维度变 [Batch, 64, 128, 128]
        x = self.res_block(x)                            # 维度仍为 [Batch, 64, 128, 128]
        x = self.pool(x)                                 # 浓缩特征：变成 [Batch, 64, 1, 1]
        x = torch.flatten(x, 1)                          # 展平：变成 [Batch, 64]
        out = self.fc(x)                                 # 最终得分：变成 [Batch, 2]
        return out

# ==========================================
# 模块 3：核心训练流水线 (GPU 咆哮的地方)
# ==========================================
def main():
    # 1. 硬件配置：探测是否有 GPU，没有就用 CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    print(f"当前使用的计算设备: {device.type.upper()}\n")

    # 2. 准备数据车队
    my_transform = transforms.Compose([
        transforms.Resize(130),         # 注意这里是一个整数。意思是保持长宽比，把较短的那条边缩放到 130 像素
        transforms.CenterCrop(128),     # 在图片正中心，像切蛋糕一样切出一个 128x128 的绝对正方形
        transforms.ToTensor()
    ])
    # train_transform = transforms.Compose([
    #     transforms.Resize(150),                              # 先稍微放大一点
    #     transforms.RandomResizedCrop(128, scale=(0.8, 1.0)), # 随机切一块 128x128 出来 (防死记硬背绝对位置)
    #     transforms.RandomHorizontalFlip(p=0.5),              # 50% 概率左右镜像翻转
    #     transforms.RandomRotation(degrees=15),               # 随机旋转 -15 到 15 度
    #     transforms.ColorJitter(brightness=0.2, contrast=0.2),# 随机稍微改变亮度和对比度
    #     transforms.ToTensor()
    # ])
    
    # # 2. 预测/测试特供：绝对原汁原味的标准流水线 (你在 predict.py 里用的就是这个)
    # test_transform = transforms.Compose([
    #     transforms.Resize(130),
    #     transforms.CenterCrop(128),
    #     transforms.ToTensor()
    # ])
    
    # # 将训练流水线装配给你的 Dataset
    # dataset = CatDogDataset(root_dir="./my_dataset", transform=train_transform)
    # 注意：这里假设你上一节生成的 my_dataset 文件夹还在当前目录下
    dataset = CatDogDataset(root_dir="./my_dataset", transform=my_transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # 3. 实例化网络、损失函数、优化器
    model = MiniResNet(num_classes=2).to(device)  # 把整个工厂搬进显卡显存
    criterion = nn.CrossEntropyLoss()             # 专门做分类题的损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.001) # Adam 是目前工业界最常用的高级优化器
    
    epochs = 50
    
    # 4. 开始炼丹 (正式训练循环)
    print("🚀 开始训练流水线...")
    # 将模型权重保存为 .pth 文件
    for epoch in range(epochs):
        model.train() # 告诉网络：现在是训练模式
        total_loss = 0.0
        
        for batch_idx, (images, labels) in enumerate(dataloader):
            # 【关键】把硬盘读到内存里的图片和标签，全部推入显卡的显存！
            images = images.to(device)
            labels = labels.to(device)
            
            # 第一步：前向传播 (跑流水线)
            outputs = model(images)
            
            # 第二步：计算误差
            loss = criterion(outputs, labels)
            
            # 第三步：反向传播与更新 (清理 -> 算导数 -> 更新参数)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1:02d}/{epochs}] | 平均 Loss: {avg_loss:.4f}")
        
    print("\n🎉 训练结束！模型已具备区分这批马赛克图像的能力。")
    torch.save(model.state_dict(), "cat_dog_model.pth")
    print("模型权重已保存至 cat_dog_model.pth")

if __name__ == '__main__':
    main()