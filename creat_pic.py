import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# ==========================================
# 1. 实战核心：自定义 ImageDataset
# ==========================================
class CatDogDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        初始化：扫描目录，记录所有的图片路径和标签
        """
        self.root_dir = root_dir
        self.transform = transform
        
        self.image_paths = []
        self.labels = []
        
        # 定义类别与数字标签的映射
        # cats -> 0, dogs -> 1
        self.class_to_idx = {"cats": 0, "dogs": 1}
        
        # 遍历根目录下的子文件夹
        for class_name in self.class_to_idx.keys():
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
                
            # 获取该类别文件夹下的所有文件名
            for file_name in os.listdir(class_dir):
                if file_name.endswith('.jpg'):
                    # 保存完整的绝对/相对路径
                    self.image_paths.append(os.path.join(class_dir, file_name))
                    # 保存对应的数字标签
                    self.labels.append(self.class_to_idx[class_name])

    def __len__(self):
        """返回数据集总共有多少张图"""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """核心：读取第 idx 张图片，进行预处理并返回"""
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # 1. 真正从硬盘读取图片 (使用 PIL 库)
        image = Image.open(img_path).convert('RGB')
        
        # 2. 如果有数据增强/转换操作，在这里执行
        if self.transform:
            image = self.transform(image)
            
        return image, label

# ==========================================
# 2. 运行测试：启动流水线
# ==========================================
if __name__ == '__main__':
    # 步骤 A: 生成数据
    # 步骤 B: 定义预处理流程 (Transforms)
    # 工业界标配：先转为 Tensor，再进行标准化归一化
    my_transform = transforms.Compose([
        transforms.Resize((128, 128)), # 强行把图片统一缩放到 128x128
        transforms.ToTensor(),         # 把 PIL 图片转为 Tensor，并将像素值从 0-255 压缩到 0-1
    ])
    
    # 步骤 C: 实例化 Dataset 和 DataLoader
    dataset = CatDogDataset(root_dir=data_dir, transform=my_transform)
    
    print(f"Dataset 成功扫描到 {len(dataset)} 张图片。")
    
    # 设置 Batch Size 为 8，打乱顺序
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # 步骤 D: 模拟训练索要数据
    for batch_idx, (images, labels) in enumerate(dataloader):
        print(f"\n--- 成功提取第 {batch_idx + 1} 个 Batch ---")
        print(f"Images Tensor 的形状: {images.shape}")
        print(f"Labels: {labels}")
        
        # 我们只看前两个 Batch 就退出
        if batch_idx == 1:
            break