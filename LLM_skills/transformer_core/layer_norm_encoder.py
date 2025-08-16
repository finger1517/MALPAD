import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm(nn.Module):
    """层归一化 - Transformer中的重要归一化方法
    
    与BatchNorm的区别：
    - BatchNorm: 对batch维度归一化，用于CNN
    - LayerNorm: 对特征维度归一化，用于Transformer
    
    实现要点：
    1. 计算均值和方差
    2. 归一化
    3. 可学习的缩放和平移参数
    
    面试重点：理解为什么Transformer使用LayerNorm而非BatchNorm
    """
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.d_model = d_model
        self.eps = eps  # 防止除零
        
        # 可学习的参数
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        
        # 计算均值和方差
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True)
        
        # 归一化
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)
        
        # 缩放和平移
        output = self.gamma * x_normalized + self.beta
        return output

class TransformerEncoderBlock(nn.Module):
    """Transformer编码器块
    
    结构：
    1. 多头自注意力 + 残差连接 + LayerNorm
    2. 前馈网络 + 残差连接 + LayerNorm
    
    面试重点：理解残差连接和LayerNorm的作用
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForwardNetwork(d_model, d_ff, dropout)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # 多头自注意力 + 残差连接 + LayerNorm
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 前馈网络 + 残差连接 + LayerNorm
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        
        return x

class FeedForwardNetwork(nn.Module):
    """前馈网络
    
    结构：Linear -> ReLU -> Dropout -> Linear
    作用：为模型引入非线性变换能力
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

# 测试代码
if __name__ == "__main__":
    from multi_head_attention import MultiHeadAttention
    
    # 测试参数
    batch_size = 2
    seq_len = 10
    d_model = 512
    num_heads = 8
    d_ff = 2048
    
    # 创建测试数据
    x = torch.randn(batch_size, seq_len, d_model)
    
    # 测试LayerNorm
    layer_norm = LayerNorm(d_model)
    output_norm = layer_norm(x)
    print(f"LayerNorm输入形状: {x.shape}")
    print(f"LayerNorm输出形状: {output_norm.shape}")
    
    # 测试Transformer编码器块
    encoder_block = TransformerEncoderBlock(d_model, num_heads, d_ff)
    output_encoder = encoder_block(x)
    print(f"Transformer编码器块输入形状: {x.shape}")
    print(f"Transformer编码器块输出形状: {output_encoder.shape}")
    
    print("LayerNorm和Transformer编码器块实现正确！")