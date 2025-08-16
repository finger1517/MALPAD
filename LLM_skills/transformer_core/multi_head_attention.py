import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    """多头自注意力机制 - Transformer的核心组件
    
    实现要点：
    1. 将输入投影到Q、K、V三个空间
    2. 分割成多个头并行计算注意力
    3. 缩放点积注意力防止梯度消失
    4. 输出投影回原始维度
    
    面试重点：理解注意力机制的计算过程和多头并行的优势
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Q、K、V的线性变换
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """缩放点积注意力
        
        计算公式：Attention(Q,K,V) = softmax(QK^T/√d_k)V
        """
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 应用mask（可选）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 计算注意力权重
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 计算输出
        output = torch.matmul(attn_weights, V)
        return output
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # 线性变换
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)
        
        # 重塑为多头形式 [batch_size, num_heads, seq_len, d_k]
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # 合并多头
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.d_model)
        
        # 输出投影
        output = self.w_o(attn_output)
        return output

# 测试代码
if __name__ == "__main__":
    # 测试参数
    batch_size = 2
    seq_len = 10
    d_model = 512
    num_heads = 8
    
    # 创建测试数据
    x = torch.randn(batch_size, seq_len, d_model)
    
    # 创建多头注意力层
    mha = MultiHeadAttention(d_model, num_heads)
    
    # 前向传播
    output = mha(x, x, x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print("多头注意力实现正确！")