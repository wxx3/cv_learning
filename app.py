import torch
import torch.nn as nn
from torchvision import transforms, models
import gradio as gr

# 1. 加载模型结构与参数 (将模型安置在内存中待命)
# 部署推理阶段通常强制使用 CPU，以节省昂贵的 GPU 资源
device = torch.device("cpu") 
model = models.resnet18()
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)

# 读取之前炼丹保存的脑子
model.load_state_dict(torch.load("cat_dog_model.pth", map_location=device, weights_only=True))
model.eval() # 极其重要：强制关闭训练时的随机 Dropout 等动态机制

# 2. 固定预处理流水线
test_transform = transforms.Compose([
    transforms.Resize(130),
    transforms.CenterCrop(128),
    transforms.ToTensor()
])

classes = ["猫 (Cat)", "狗 (Dog)"]

# 3. 编写业务逻辑接口
def predict_image(img):
    # img 是网页前端传进来的 PIL 图像对象
    img_tensor = test_transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(img_tensor) # 得到原始得分，如 [[3.2, -1.5]]
        
        # 物理操作：使用 Softmax 函数，将原始得分按自然对数 e 的指数放大后，计算占比
        # 这会将任意数字强行转换为 0.0 到 1.0 之间，且两项加起来绝对等于 1 的概率分布
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        
    # 组装返回给前端的数据结构：{ "猫 (Cat)": 0.85, "狗 (Dog)": 0.15 }
    return {classes[i]: float(probabilities[i]) for i in range(2)}

# 4. 构建并启动网页 UI
demo = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil", label="上传猫或狗的图片"),
    outputs=gr.Label(num_top_classes=2, label="AI 预测概率"),
    title="视觉分类模型部署测试",
    description="上传图片，浏览器将直接调用本地的 cat_dog_model.pth 进行实时推演。"
)

if __name__ == "__main__":
    demo.launch()