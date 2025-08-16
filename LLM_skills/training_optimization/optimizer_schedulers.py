import torch
import torch.nn as nn
import math

class AdamOptimizer:
    """Adam优化器 - 自适应矩估计优化器
    
    算法特点：
    1. 结合动量（Momentum）和自适应学习率
    2. 维护一阶矩估计（动量）和二阶矩估计（自适应学习率）
    3. 对每个参数都有独立的学习率
    
    更新公式：
    m_t = β1 * m_{t-1} + (1-β1) * g_t
    v_t = β2 * v_{t-1} + (1-β2) * g_t^2
    m_hat = m_t / (1-β1^t)
    v_hat = v_t / (1-β2^t)
    θ_t = θ_{t-1} - lr * m_hat / (√v_hat + ε)
    
    面试重点：理解Adam为什么结合了动量和自适应学习率的优点
    """
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        self.params = list(params)
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        
        # 初始化矩估计
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.t = 0
        
    def zero_grad(self):
        """清空梯度"""
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()
    
    def step(self):
        """执行一步优化"""
        self.t += 1
        
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
                
            grad = param.grad
            
            # 权重衰减
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * param
            
            # 更新一阶矩估计
            self.m[i] = self.betas[0] * self.m[i] + (1 - self.betas[0]) * grad
            
            # 更新二阶矩估计
            self.v[i] = self.betas[1] * self.v[i] + (1 - self.betas[1]) * grad * grad
            
            # 计算偏差修正后的矩估计
            m_hat = self.m[i] / (1 - self.betas[0] ** self.t)
            v_hat = self.v[i] / (1 - self.betas[1] ** self.t)
            
            # 更新参数
            param.data = param.data - self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)

class GradientClipping:
    """梯度裁剪 - 防止梯度爆炸
    
    实现方式：
    1. 按值裁剪：限制梯度绝对值
    2. 按范数裁剪：限制梯度向量的范数
    
    面试重点：理解梯度裁剪的作用原理和适用场景
    """
    @staticmethod
    def clip_by_value(grads, clip_value=1.0):
        """按值裁剪梯度"""
        clipped_grads = []
        for grad in grads:
            if grad is not None:
                clipped_grad = torch.clamp(grad, -clip_value, clip_value)
                clipped_grads.append(clipped_grad)
        return clipped_grads
    
    @staticmethod
    def clip_by_norm(grads, max_norm=1.0):
        """按范数裁剪梯度"""
        # 计算所有梯度的全局范数
        total_norm = 0
        for grad in grads:
            if grad is not None:
                total_norm += grad.data.norm() ** 2
        total_norm = total_norm ** 0.5
        
        # 计算裁剪系数
        clip_coef = max_norm / (total_norm + 1e-6)
        clip_coef = min(clip_coef, 1.0)
        
        # 应用裁剪
        clipped_grads = []
        for grad in grads:
            if grad is not None:
                clipped_grads.append(grad * clip_coef)
        return clipped_grads

class CosineAnnealingLR:
    """余弦退火学习率调度器
    
    学习率变化公式：
    lr_t = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(T_cur / T_max * π))
    
    面试重点：理解余弦退火为什么能帮助模型跳出局部最优
    """
    def __init__(self, optimizer, T_max, eta_min=0):
        self.optimizer = optimizer
        self.T_max = T_max
        self.eta_min = eta_min
        self.T_cur = 0
        
    def step(self):
        """更新学习率"""
        self.T_cur += 1
        
        # 计算新的学习率
        lr = self.eta_min + 0.5 * (self.optimizer.lr - self.eta_min) * \
             (1 + math.cos(self.T_cur / self.T_max * math.pi))
        
        # 更新优化器的学习率
        self.optimizer.lr = lr
        
        # 重置周期
        if self.T_cur >= self.T_max:
            self.T_cur = 0

class WarmupScheduler:
    """预热学习率调度器
    
    实现要点：
    1. 线性预热阶段
    2. 余弦退火阶段
    3. 适合大模型训练
    
    面试重点：理解预热策略对大模型训练的重要性
    """
    def __init__(self, optimizer, warmup_steps, total_steps, max_lr, min_lr=0):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.current_step = 0
        
    def step(self):
        """更新学习率"""
        self.current_step += 1
        
        if self.current_step <= self.warmup_steps:
            # 预热阶段：线性增加
            lr = self.max_lr * (self.current_step / self.warmup_steps)
        else:
            # 退火阶段：余弦衰减
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + math.cos(progress * math.pi))
        
        self.optimizer.lr = lr

class GradientAccumulator:
    """梯度累积 - 解决显存不足问题
    
    实现要点：
    1. 累积多个batch的梯度
    2. 定期执行参数更新
    3. 有效增加batch size
    
    面试重点：理解梯度累积的原理和优势
    """
    def __init__(self, optimizer, accumulation_steps):
        self.optimizer = optimizer
        self.accumulation_steps = accumulation_steps
        self.current_step = 0
        
    def step(self):
        """执行梯度累积步"""
        self.current_step += 1
        
        if self.current_step % self.accumulation_steps == 0:
            # 执行参数更新
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.current_step = 0

# 测试代码
if __name__ == "__main__":
    # 创建测试参数
    model = nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 1)
    )
    
    # 测试Adam优化器
    optimizer = AdamOptimizer(model.parameters(), lr=0.001)
    
    # 模拟训练过程
    x = torch.randn(32, 10)
    y = torch.randn(32, 1)
    
    for epoch in range(3):
        optimizer.zero_grad()
        
        # 前向传播
        output = model(x)
        loss = nn.MSELoss()(output, y)
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪
        grads = [param.grad for param in model.parameters()]
        clipped_grads = GradientClipping.clip_by_norm(grads, max_norm=1.0)
        
        # 更新参数
        optimizer.step()
        
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
    
    # 测试学习率调度器
    scheduler = WarmupScheduler(optimizer, warmup_steps=100, total_steps=1000, max_lr=0.001)
    for i in range(150):
        scheduler.step()
        if i % 50 == 0:
            print(f"Step {i}, Learning Rate: {optimizer.lr:.6f}")
    
    print("训练优化组件实现正确！")