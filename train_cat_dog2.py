import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models # 【新增】导入 PyTorch 官方模型库
from PIL import Image

# ==========================================
# 模块 1：数据读取 (数据工程车间)
# ==========================================
class CatDogDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_paths, self.labels = [], []
        self.transform = transform
        class_to_idx = {"cats": 0, "dogs": 1} # 铁律：猫=0，狗=1
        
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
        # 【物理操作】硬盘磁头转动，把这张 .jpg 读入内存，并交给 transform 流水线去裁剪变形
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]

# ==========================================
# 模块 2：核心训练流水线 (GPU 开始咆哮)
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前使用的计算设备: {device.type.upper()}\n")

    # 1. 魔法数据增强 (只在训练时用)
    train_transform = transforms.Compose([
        transforms.Resize(150),                              # 先稍微放大
        transforms.RandomResizedCrop(128, scale=(0.8, 1.0)), # 随机挖一块出来，防止模型死记硬背
        transforms.RandomHorizontalFlip(p=0.5),              # 50%的概率镜像翻转，猫脸朝左变朝右
        transforms.RandomRotation(degrees=15),               # 随机歪头
        transforms.ColorJitter(brightness=0.2, contrast=0.2),# 随机改变光照条件
        transforms.ToTensor()                                # 变成显卡认识的张量 (Tensor)
    ])
    
    # 2. 准备数据车队
    dataset = CatDogDataset(root_dir="./my_dataset", transform=train_transform)
    # num_workers=0 是因为 Windows 下多线程容易报错，如果你想加速，可以尝试改成 2 或 4
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # ==========================================
    # 【高能预警】外科手术开始：准备迁移学习
    # ==========================================
    print("正在下载/加载 ResNet18 视觉博士的脑子...")
    # 第一步：直接向官方服务器请求一个在 120 万张图片上训练好的 ResNet18
    # 它的脑子里自带了几百个神奇的卷积核，懂得看猫耳朵、狗毛、轮胎、眼睛...
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # 第二步：打麻药，冻结“博士”的大脑。
    # 为什么要冻结？因为我们的猫狗数据太少，如果让这部分跟着一起训，
    # 很容易把博士原本聪明的脑子给“带偏了”（这叫灾难性遗忘）。
    for param in model.parameters():
        param.requires_grad = False # 告诉显卡：这部分不用算导数了，省去 90% 的计算量！

    # 第三步：换嘴巴。
    # 博士本来能认识 1000 种东西（默认的输出层是 nn.Linear(512, 1000)）
    # 我们把他的旧嘴巴拆下来，换上一个新的、只有 2 个输出孔（猫、狗）的新嘴巴。
    num_features = model.fc.in_features # 获取原来输入这根管道的粗细（512）
    model.fc = nn.Linear(num_features, 2) # 接上新的输出管
    # 注意：新换上的这块肉（model.fc）是随机初始化的，没有任何记忆，所以它的 requires_grad 默认是 True。
    
    # 把做完手术的模型搬进显卡
    model = model.to(device) 

    # 3. 准备打分器和教练
    criterion = nn.CrossEntropyLoss()
    
    # 【极其关键的一步】：
    # 以前我们写的是 model.parameters()，意思是整个模型都归教练管。
    # 现在的教练（优化器）只负责训练“新嘴巴（model.fc.parameters()）”。脑子的其余部分归麻醉师管，教练不碰。
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001) 
    
    epochs = 50
    
    # ==========================================
    # 4. 正式炼丹
    # ==========================================
    print("🚀 开始极速微调流水线...")
    for epoch in range(epochs):
        model.train() 
        total_loss = 0.0
        
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.to(device)
            
            # 第一步：前向传播（跑完整个 ResNet18）
            outputs = model(images)
            
            # 第二步：算误差
            loss = criterion(outputs, labels)
            
            # 第三步：清理上一次的垃圾 -> 算新导数 -> 更新参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1:02d}/{epochs}] | 平均 Loss: {avg_loss:.4f}")
        
    print("\n🎉 训练结束！")
    
    # 5. 【修复的 Bug】等 50 轮全部跑完，再把聪明的脑子存入硬盘！
    torch.save(model.state_dict(), "cat_dog_model.pth")
    print("模型权重已保存至 cat_dog_model.pth")

if __name__ == '__main__':
    main()