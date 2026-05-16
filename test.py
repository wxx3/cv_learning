import numpy as np
import matplotlib.pyplot as plt # 仅用于可视化结果

# ==========================================
# 1. 生成模拟数据
# ==========================================
np.random.seed(42)
N = 100       # 样本数量
d = 1         # 输入特征维度
m = 1         # 输出特征维度

# 生成 X，形状 (100, 1)
X = 2 * np.random.rand(N, d)
# 生成真实的 Y，假设真实权重为 3，偏置为 4，并加入高斯噪声
Y_true = 3 * X + 4 + np.random.randn(N, m) * 0.5

# ==========================================
# 2. 初始化模型参数
# ==========================================
# 权重 W 形状 (1, 1)，偏置 b 形状 (1,)
W = np.random.randn(d, m)
b = np.random.randn(m)

learning_rate = 0.1
epochs = 100

# ==========================================
# 3. 训练循环 (Training Loop) - 工业界标准结构
# ==========================================
for epoch in range(epochs):
    # --- 前向传播 (Forward Pass) ---
    # 对应公式：Y_pred = X * W + b
    # 预测是吗
    Y_pred = np.dot(X, W) + b
    # 计算损失 (MSE)#与真实值的偏差
    error = Y_pred - Y_true
    
    loss = np.mean(error ** 2)
    
    # --- 反向传播 (Backward Pass) ---
    # 对应公式：dW = (2/N) * X^T * (Y_pred - Y_true)
    dW = (2 / N) * np.dot(X.T, error)
    # db 则是误差的平均值
    db = (2 / N) * np.sum(error)
    
    # --- 梯度更新 (Parameter Update) ---
    W -= learning_rate * dW
    b -= learning_rate * db
    
    # 打印日志
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:03d}/{epochs} | Loss: {loss:.4f} | W: {W[0][0]:.4f} | b: {b[0]:.4f}")

# ==========================================
# 4. 验证结果
# ==========================================
print("\n训练结束。")
print(f"预测参数: W = {W[0][0]:.4f}, b = {b[0]:.4f}")