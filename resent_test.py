import torch
import torch.nn as nn

# ==========================================
# 1. 定义 ResNet 的核心：残差块 (Basic Block)
# ==========================================
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # 第一层卷积：提取图像特征
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels) # 批归一化，让训练更稳定
        self.relu = nn.ReLU(inplace=True)       # 激活函数
        
        # 第二层卷积：进一步提取特征
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 如果输入和输出的通道数不一样，需要一个额外的 1x1 卷积来对齐维度
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        # 1. 保存原始输入 (这是残差网络的灵魂！)
        identity = self.shortcut(x)
        
        # 2. 正常走两层卷积
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # 3. 【重点】将卷积后的结果与原始输入相加！(F(x) + x)
        out += identity
        
        # 4. 最后再过一次激活函数
        out = self.relu(out)
        
        return out

# ==========================================
# 2. 模拟从上一节 DataLoader 出来的 Batch 数据
# ==========================================
# 假设这是我们刚才那 8 张 128x128 的彩色马赛克图
# 维度：[Batch_size=8, Channels=3, Height=128, Width=128]
dummy_images = torch.randn(8, 3, 128, 128)

print(f"1. 输入网络的张量形状: {dummy_images.shape}")

# ==========================================
# 3. 让数据流过网络
# ==========================================
# 实例化一个残差块：让输入的 3 个颜色通道变成 64 个特征通道
res_block = ResidualBlock(in_channels=3, out_channels=64)

# 执行前向传播
output_tensor = res_block(dummy_images)

print(f"2. 经过 ResNet 核心块后的张量形状: {output_tensor.shape}")