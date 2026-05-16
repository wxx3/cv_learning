import torch
from torchvision import transforms
from PIL import Image
from train_cat_dog import MiniResNet # 从训练脚本导入你写的网络结构

def predict_single_image(image_path, model_path):
    # 1. 准备相同的特征提取器（空模型）
    model = MiniResNet(num_classes=2)
    
    # 2. 注入灵魂（加载训练好的参数）
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval() # 切换到评估模式，关闭 Dropout 和 BatchNorm 的动态更新
    
    # 3. 准备和训练时一模一样的数据预处理流水线
    transform = transforms.Compose([
        transforms.Resize(130),         # 注意这里是一个整数。意思是保持长宽比，把较短的那条边缩放到 130 像素
        transforms.CenterCrop(128),     # 在图片正中心，像切蛋糕一样切出一个 128x128 的绝对正方形
        transforms.ToTensor()
    ])
    
    # 4. 读取图片并增加 Batch 维度 (从 [3, 128, 128] 变成 [1, 3, 128, 128])
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)
    
    # 5. 前向传播，输出得分
    with torch.no_grad(): # 明确告诉 PyTorch：现在是使用阶段，不需要算梯度，省点内存
        output = model(img_tensor)
        
        # 提取类别索引
        _, predicted_idx = torch.max(output, 1)
        
    classes = ["cats", "dogs"]
    print(f"原始网络输出得分: {output.numpy()}")
    print(f"最终预测结果: {classes[predicted_idx.item()]}")

if __name__ == '__main__':
    # 随便找一张你 my_dataset 里的图片路径去测试
    test_img = ".\my_dataset\dogs\dogs_0.jpg"
    predict_single_image(test_img, "cat_dog_model.pth")