import torch
import torch.nn as nn
import torch.optim as optim

# ==========================================
# 1. 数据准备 (与之前一致，但必须转为 Tensor)
# ==========================================
# 生成同样的模拟数据，转为 float32 格式（PyTorch 默认精度）
X_np = 2 * torch.rand(100, 1)
Y_true_np = 3 * X_np + 4 + torch.randn(100, 1) * 0.5

# ==========================================
# 2. 定义模型、损失函数和优化器
# ==========================================
# nn.Linear(1, 1) 等同于 Y = XW + b，自动初始化 W 和 b，并接管它们的维度
model = nn.Linear(in_features=1, out_features=1)

# MSELoss 就是均方误差公式：np.mean((Y_pred - Y_true)**2)
criterion = nn.MSELoss()

# SGD 就是梯度下降，接管模型中所有需要更新的参数，设置学习率为 0.1
optimizer = optim.SGD(model.parameters(), lr=0.1)

epochs = 100

# ==========================================
# 3. 训练循环 (工业级标准写法)
# ==========================================
for epoch in range(epochs):
    # --- 1. 前向传播 ---
    Y_pred = model(X_np) #用模型创建了算出了y的预测值，
    #建立了一个链 w, b -> y_pred
    
    # --- 2. 计算损失 ---
    loss = criterion(Y_pred, Y_true_np) #用y_pred计算了损失
    #建立了 w, b -> y_pred -> loss
    
    # --- 3. 反向传播 (自动求导) ---
    # 【重点】清除上一轮残留的梯度
    optimizer.zero_grad() 
    # 【重点】这一步自动帮你计算出了 dW 和 db，并存储在参数的 .grad 属性中
    loss.backward()
    # 顺着链路去反向更新了dw和db。

    # --- 4. 梯度更新 ---
    # 这一步等于：W -= learning_rate * dW 和 b -= learning_rate * db
    optimizer.step()
    
    # 打印日志（loss.item() 用于从 Tensor 中提取标量数值）
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:03d}/{epochs} | Loss: {loss.item():.4f}")

# ==========================================
# 4. 验证结果
# ==========================================
# 提取模型内部的参数 W (weight) 和 b (bias)
predicted_W = model.weight.item()
predicted_b = model.bias.item()

print("\n训练结束。")
print(f"真实参数: W = 3, b = 4")
print(f"预测参数: W = {predicted_W:.4f}, b = {predicted_b:.4f}")