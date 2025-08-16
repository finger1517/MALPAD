import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """位置编码 - 为Transformer提供序列位置信息
    
    实现要点：
    1. 使用正弦和余弦函数生成位置编码
    2. 偶数位置使用sin，奇数位置使用cos
    3. 不同频率的波形编码不同位置
    4. 相对位置信息通过线性变换可学习
    
    面试重点：理解为什么使用三角函数而非可学习参数
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.d_model = d_model
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # 计算div_term：10000^(2i/d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        # 计算位置编码
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维度用sin
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维度用cos
        
        # 注册为buffer，不参与训练
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        # 将位置编码加到输入上
        return x + self.pe[:, :x.size(1)]

class LearnablePositionalEncoding(nn.Module):
    """可学习的位置编码
    
    与固定位置编码的对比：
    - 优点：可以适应特定任务
    - 缺点：泛化能力较差，难以处理超出训练长度的序列
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model))
        nn.init.xavier_uniform_(self.pos_embedding)
        
    def forward(self, x):
        return x + self.pos_embedding[:, :x.size(1)]

# 测试代码
if __name__ == "__main__":
    # 测试参数
    batch_size = 2
    seq_len = 10
    d_model = 512
    
    # 创建测试数据
    x = torch.randn(batch_size, seq_len, d_model)
    
    # 测试固定位置编码
    fixed_pe = PositionalEncoding(d_model)
    output_fixed = fixed_pe(x)
    
    # 测试可学习位置编码
    learnable_pe = LearnablePositionalEncoding(d_model)
    output_learnable = learnable_pe(x)
    
    print(f"输入形状: {x.shape}")
    print(f"固定位置编码输出形状: {output_fixed.shape}")
    print(f"可学习位置编码输出形状: {output_learnable.shape}")
    print("位置编码实现正确！")
    
    # 可视化位置编码（可选）
    import matplotlib.pyplot as plt
    pe_matrix = fixed_pe.pe.squeeze(0).numpy()
    plt.figure(figsize=(12, 8))
    plt.imshow(pe_matrix[:20, :20], cmap='viridis')
    plt.colorbar()
    plt.title("Positional Encoding Visualization")
    plt.xlabel("Embedding Dimension")
    plt.ylabel("Position")
    plt.savefig("positional_encoding.png")
    print("位置编码可视化已保存！")