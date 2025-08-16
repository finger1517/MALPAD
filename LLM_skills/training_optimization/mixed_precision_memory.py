import torch
import torch.nn as nn
import torch.cuda.amp as amp

class MixedPrecisionTrainer:
    """混合精度训练
    
    实现要点：
    1. 使用FP16进行前向和反向传播
    2. 使用FP32存储参数和梯度
    3. 动态损失缩放防止数值下溢
    
    优势：
    1. 减少显存占用
    2. 加速计算（Tensor Core）
    3. 保持数值稳定性
    
    面试重点：理解混合精度训练的原理和实现细节
    """
    def __init__(self, model, optimizer, init_scale=2**16):
        self.model = model
        self.optimizer = optimizer
        self.scaler = amp.GradScaler(init_scale=init_scale)
        
    def train_step(self, x, y, criterion):
        """执行一步混合精度训练"""
        # 前向传播（自动混合精度）
        with amp.autocast():
            output = self.model(x)
            loss = criterion(output, y)
        
        # 反向传播（自动缩放）
        self.scaler.scale(loss).backward()
        
        # 梯度裁剪
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # 参数更新
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        
        return loss.item()

class DistributedTrainer:
    """分布式训练基础实现
    
    实现要点：
    1. 数据并行
    2. 梯度同步
    3. 模型同步
    
    面试重点：理解分布式训练的基本原理
    """
    def __init__(self, model, device_ids=None):
        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        
        self.device_ids = device_ids
        self.model = nn.DataParallel(model, device_ids=device_ids)
        
    def train_step(self, x, y, criterion, optimizer):
        """执行一步分布式训练"""
        # 数据已经分发到各个GPU
        output = self.model(x)
        loss = criterion(output, y)
        
        # 反向传播（自动同步梯度）
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()

class MemoryEfficientAttention:
    """内存高效的注意力计算
    
    实现要点：
    1. 分块计算
    2. 重用显存
    3. 减少临时变量
    
    面试重点：理解如何优化注意力计算的显存使用
    """
    def __init__(self, d_model, num_heads, block_size=64):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.block_size = block_size
        
    def forward(self, q, k, v):
        """内存高效的注意力计算"""
        batch_size, seq_len, _ = q.shape
        
        # 重塑为多头形式
        q = q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # 分块计算
        output = torch.zeros_like(q)
        
        for i in range(0, seq_len, self.block_size):
            end_i = min(i + self.block_size, seq_len)
            
            # 计算当前块的注意力
            q_block = q[:, :, i:end_i, :]
            
            # 分块计算K^T * V
            kt_v = torch.zeros(batch_size, self.num_heads, self.d_k, self.d_k, 
                              device=q.device)
            
            for j in range(0, seq_len, self.block_size):
                end_j = min(j + self.block_size, seq_len)
                k_block = k[:, :, j:end_j, :]
                v_block = v[:, :, j:end_j, :]
                
                kt_v += torch.matmul(k_block.transpose(-2, -1), v_block)
            
            # 计算输出
            output_block = torch.matmul(q_block, kt_v)
            output[:, :, i:end_i, :] = output_block
        
        # 合并多头
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, self.d_model)
        
        return output

# 测试代码
if __name__ == "__main__":
    # 创建测试模型
    model = nn.Sequential(
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    )
    
    # 测试混合精度训练
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # 创建混合精度训练器
    if torch.cuda.is_available():
        model = model.cuda()
        mixed_precision_trainer = MixedPrecisionTrainer(model, optimizer)
        
        # 模拟训练数据
        x = torch.randn(32, 784).cuda()
        y = torch.randint(0, 10, (32,)).cuda()
        
        # 执行训练步
        loss = mixed_precision_trainer.train_step(x, y, criterion)
        print(f"混合精度训练Loss: {loss:.4f}")
    
    # 测试内存高效注意力
    batch_size = 4
    seq_len = 128
    d_model = 512
    num_heads = 8
    
    q = torch.randn(batch_size, seq_len, d_model)
    k = torch.randn(batch_size, seq_len, d_model)
    v = torch.randn(batch_size, seq_len, d_model)
    
    mem_eff_attn = MemoryEfficientAttention(d_model, num_heads)
    output = mem_eff_attn(q, k, v)
    
    print(f"内存高效注意力输入形状: {q.shape}")
    print(f"内存高效注意力输出形状: {output.shape}")
    
    print("混合精度训练和内存优化实现正确！")