import torch
import torch.nn as nn
import torch.nn.functional as F
from multi_head_attention import MultiHeadAttention
from layer_norm_encoder import LayerNorm, FeedForwardNetwork

class TransformerDecoderBlock(nn.Module):
    """Transformer解码器块
    
    结构：
    1. 掩码多头自注意力 + 残差连接 + LayerNorm
    2. 编码器-解码器注意力 + 残差连接 + LayerNorm
    3. 前馈网络 + 残差连接 + LayerNorm
    
    面试重点：理解解码器的两种注意力机制和掩码的作用
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        # 掩码自注意力（只关注当前位置之前的内容）
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        # 编码器-解码器注意力（关注编码器的输出）
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        
        self.ffn = FeedForwardNetwork(d_model, d_ff, dropout)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, tgt_mask=None, memory_mask=None):
        # 掩码自注意力 + 残差连接 + LayerNorm
        self_attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_output))
        
        # 编码器-解码器注意力 + 残差连接 + LayerNorm
        cross_attn_output = self.cross_attn(x, enc_output, enc_output, memory_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))
        
        # 前馈网络 + 残差连接 + LayerNorm
        ffn_output = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_output))
        
        return x

def create_causal_mask(seq_len):
    """创建因果掩码（后续掩码）
    
    作用：确保解码器只能关注当前位置之前的内容
    """
    mask = torch.tril(torch.ones(seq_len, seq_len))
    return mask.unsqueeze(0).unsqueeze(0)

def create_padding_mask(seq, pad_idx=0):
    """创建填充掩码
    
    作用：忽略padding位置的注意力计算
    """
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)

# 测试代码
if __name__ == "__main__":
    # 测试参数
    batch_size = 2
    seq_len = 10
    d_model = 512
    num_heads = 8
    d_ff = 2048
    
    # 创建测试数据
    x = torch.randn(batch_size, seq_len, d_model)
    enc_output = torch.randn(batch_size, seq_len, d_model)
    
    # 创建掩码
    causal_mask = create_causal_mask(seq_len)
    
    # 测试Transformer解码器块
    decoder_block = TransformerDecoderBlock(d_model, num_heads, d_ff)
    output_decoder = decoder_block(x, enc_output, causal_mask)
    
    print(f"解码器输入形状: {x.shape}")
    print(f"编码器输出形状: {enc_output.shape}")
    print(f"解码器输出形状: {output_decoder.shape}")
    print("Transformer解码器块实现正确！")
    
    # 测试掩码功能
    print("因果掩码形状:", causal_mask.shape)
    print("因果掩码示例（前5x5）:")
    print(causal_mask[0, 0, :5, :5])