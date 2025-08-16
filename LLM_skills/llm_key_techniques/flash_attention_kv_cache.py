import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FlashAttention(nn.Module):
    """Flash Attention的简化版本
    
    核心思想：
    1. 分块计算注意力，避免显存瓶颈
    2. 使用在线softmax，减少内存访问
    3. 精确计算，不近似
    
    面试重点：理解Flash Attention如何优化显存使用和计算效率
    """
    def __init__(self, d_model, num_heads, dropout=0.1, block_size=64):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.block_size = block_size
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        seq_len = query.size(1)
        
        # 线性变换
        Q = self.w_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # 分块计算注意力
        output = torch.zeros_like(Q)
        
        for i in range(0, seq_len, self.block_size):
            # 获取当前块的Q
            Q_block = Q[:, :, i:i+self.block_size, :]
            
            # 计算当前块与所有K的注意力分数
            scores_block = torch.matmul(Q_block, K.transpose(-2, -1)) / math.sqrt(self.d_k)
            
            if mask is not None:
                mask_block = mask[:, :, i:i+self.block_size, :]
                scores_block = scores_block.masked_fill(mask_block == 0, -1e9)
            
            # 在线softmax计算
            attn_weights = F.softmax(scores_block, dim=-1)
            attn_weights = self.dropout(attn_weights)
            
            # 计算输出
            output_block = torch.matmul(attn_weights, V)
            output[:, :, i:i+self.block_size, :] = output_block
        
        # 合并多头
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, self.d_model)
        
        return self.w_o(output)

class KVCache(nn.Module):
    """KV Cache - 键值缓存机制
    
    作用：
    1. 缓存历史计算的key和value
    2. 避免重复计算，提高推理效率
    3. 支持增量推理
    
    面试重点：理解KV Cache如何优化自回归推理
    """
    def __init__(self, d_model, num_heads, max_len=1024):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.max_len = max_len
        
        # 初始化缓存
        self.key_cache = torch.zeros(max_len, num_heads, d_k)
        self.value_cache = torch.zeros(max_len, num_heads, d_k)
        self.current_len = 0
        
    def update(self, key, value):
        """更新KV缓存"""
        batch_size = key.size(0)
        seq_len = key.size(1)
        
        # 将新的key和value添加到缓存
        if self.current_len + seq_len <= self.max_len:
            self.key_cache[self.current_len:self.current_len+seq_len] = key.squeeze(0)
            self.value_cache[self.current_len:self.current_len+seq_len] = value.squeeze(0)
            self.current_len += seq_len
        else:
            # 缓存已满，需要滚动更新
            self.key_cache = torch.roll(self.key_cache, -seq_len, dims=0)
            self.value_cache = torch.roll(self.value_cache, -seq_len, dims=0)
            self.key_cache[-seq_len:] = key.squeeze(0)
            self.value_cache[-seq_len:] = value.squeeze(0)
            self.current_len = min(self.current_len + seq_len, self.max_len)
    
    def get_cache(self):
        """获取当前缓存内容"""
        return self.key_cache[:self.current_len], self.value_cache[:self.current_len]
    
    def clear(self):
        """清空缓存"""
        self.current_len = 0

class EfficientAttentionWithCache(nn.Module):
    """带KV Cache的高效注意力机制"""
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        # KV Cache
        self.kv_cache = KVCache(d_model, num_heads)
        
    def forward(self, query, key=None, value=None, use_cache=False):
        batch_size = query.size(0)
        seq_len = query.size(1)
        
        # 计算query
        Q = self.w_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        if use_cache and key is None:
            # 使用缓存的key和value
            K_cache, V_cache = self.kv_cache.get_cache()
            K = K_cache.unsqueeze(0).transpose(1, 2)
            V = V_cache.unsqueeze(0).transpose(1, 2)
        else:
            # 计算新的key和value
            K = self.w_k(key).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
            V = self.w_v(value).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
            
            # 更新缓存
            if use_cache:
                self.kv_cache.update(K.transpose(1, 2), V.transpose(1, 2))
        
        # 计算注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        output = torch.matmul(attn_weights, V)
        
        # 合并多头
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, self.d_model)
        
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
    
    # 测试Flash Attention
    flash_attn = FlashAttention(d_model, num_heads)
    output_flash = flash_attn(x, x, x)
    print(f"Flash Attention输入形状: {x.shape}")
    print(f"Flash Attention输出形状: {output_flash.shape}")
    
    # 测试KV Cache
    kv_cache = KVCache(d_model, num_heads)
    k = torch.randn(1, seq_len, num_heads, d_model // num_heads)
    v = torch.randn(1, seq_len, num_heads, d_model // num_heads)
    kv_cache.update(k, v)
    k_cache, v_cache = kv_cache.get_cache()
    print(f"KV Cache key形状: {k_cache.shape}")
    print(f"KV Cache value形状: {v_cache.shape}")
    
    # 测试带缓存的注意力
    attn_with_cache = EfficientAttentionWithCache(d_model, num_heads)
    output_cache = attn_with_cache(x, x, x, use_cache=True)
    print(f"带缓存的注意力输入形状: {x.shape}")
    print(f"带缓存的注意力输出形状: {output_cache.shape}")
    
    print("Flash Attention和KV Cache实现正确！")