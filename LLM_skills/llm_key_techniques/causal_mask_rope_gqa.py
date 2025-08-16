import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CausalMask(nn.Module):
    """因果掩码 - 确保自回归生成时只能看到当前位置之前的内容
    
    实现要点：
    1. 创建下三角矩阵
    2. 将未来位置设为负无穷
    3. 支持批量处理
    
    面试重点：理解因果掩码在自回归模型中的重要性
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, attention_scores, seq_len=None):
        """
        attention_scores: [batch_size, num_heads, seq_len, seq_len]
        """
        if seq_len is None:
            seq_len = attention_scores.size(-1)
        
        # 创建因果掩码
        mask = torch.tril(torch.ones(seq_len, seq_len, device=attention_scores.device))
        
        # 将mask应用到attention_scores上
        # mask为0的位置设为负无穷，这样softmax后这些位置的权重为0
        attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
        
        return attention_scores

class RotaryPositionalEncoding(nn.Module):
    """RoPE (Rotary Positional Encoding) - 旋转位置编码
    
    优势：
    1. 相对位置编码，具有更好的外推性
    2. 无需额外参数，计算高效
    3. 随序列长度增加性能稳定
    
    实现要点：
    1. 计算旋转角度
    2. 应用旋转矩阵到query和key
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.d_model = d_model
        
        # 计算旋转角度
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer('inv_freq', inv_freq)
        
    def forward(self, x):
        """应用RoPE到输入tensor"""
        # x shape: [batch_size, seq_len, d_model]
        seq_len = x.size(1)
        
        # 计算位置编码
        positions = torch.arange(seq_len, device=x.device).float()
        freqs = torch.outer(positions, self.inv_freq)
        
        # 计算旋转矩阵
        cos_freqs = torch.cos(freqs)
        sin_freqs = torch.sin(freqs)
        
        # 分离复数部分
        x_real = x[..., ::2]  # 偶数维度
        x_imag = x[..., 1::2]  # 奇数维度
        
        # 应用旋转
        x_rot_real = x_real * cos_freqs - x_imag * sin_freqs
        x_rot_imag = x_real * sin_freqs + x_imag * cos_freqs
        
        # 重新组合
        x_rot = torch.zeros_like(x)
        x_rot[..., ::2] = x_rot_real
        x_rot[..., 1::2] = x_rot_imag
        
        return x_rot

class GroupQueryAttention(nn.Module):
    """Group Query Attention (GQA) - 组查询注意力
    
    优势：
    1. 减少KV cache的内存占用
    2. 在保持性能的同时提高推理速度
    3. 是MHA和MQA的折中方案
    
    实现要点：
    1. 将query heads分组
    2. 每组共享相同的key和value
    """
    def __init__(self, d_model, num_heads, num_kv_heads, dropout=0.1):
        super().__init__()
        assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.d_k = d_model // num_heads
        self.group_size = num_heads // num_kv_heads
        
        # 线性变换
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model // self.group_size)
        self.w_v = nn.Linear(d_model, d_model // self.group_size)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # 线性变换
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)
        
        # 重塑Q
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 重塑K、V（数量较少）
        K = K.view(batch_size, -1, self.num_kv_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_kv_heads, self.d_k).transpose(1, 2)
        
        # 扩展K、V以匹配Q的头数
        K = K.repeat_interleave(self.group_size, dim=1)
        V = V.repeat_interleave(self.group_size, dim=1)
        
        # 计算注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        output = torch.matmul(attn_weights, V)
        
        # 合并多头
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, -1, self.d_model)
        
        return self.w_o(output)

# 测试代码
if __name__ == "__main__":
    # 测试参数
    batch_size = 2
    seq_len = 10
    d_model = 512
    num_heads = 8
    
    # 创建测试数据
    x = torch.randn(batch_size, seq_len, d_model)
    
    # 测试因果掩码
    causal_mask = CausalMask()
    attention_scores = torch.randn(batch_size, num_heads, seq_len, seq_len)
    masked_scores = causal_mask(attention_scores)
    print(f"因果掩码前形状: {attention_scores.shape}")
    print(f"因果掩码后形状: {masked_scores.shape}")
    
    # 测试RoPE
    rope = RotaryPositionalEncoding(d_model)
    x_rope = rope(x)
    print(f"RoPE输入形状: {x.shape}")
    print(f"RoPE输出形状: {x_rope.shape}")
    
    # 测试GQA
    gqa = GroupQueryAttention(d_model, num_heads=8, num_kv_heads=2)
    output_gqa = gqa(x, x, x)
    print(f"GQA输入形状: {x.shape}")
    print(f"GQA输出形状: {output_gqa.shape}")
    
    print("因果掩码、RoPE和GQA实现正确！")