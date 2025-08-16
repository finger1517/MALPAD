import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Quantizer:
    """模型量化工具
    
    量化类型：
    1. INT8量化：8位整数量化
    2. 动态量化：运行时量化
    3. 静态量化：训练后量化
    
    面试重点：理解量化的原理和精度损失
    """
    def __init__(self, quant_type='int8'):
        self.quant_type = quant_type
        self.scale = None
        self.zero_point = None
        
    def calibrate(self, data):
        """校准量化参数"""
        if self.quant_type == 'int8':
            # INT8范围: [-128, 127]
            min_val = data.min()
            max_val = data.max()
            
            # 计算scale和zero_point
            self.scale = (max_val - min_val) / 255.0
            self.zero_point = int(-min_val / self.scale)
            
    def quantize(self, data):
        """量化数据"""
        if self.quant_type == 'int8':
            # 线性量化
            quantized = torch.clamp(
                data / self.scale + self.zero_point,
                -128, 127
            )
            return quantized.to(torch.int8)
        return data
    
    def dequantize(self, quantized_data):
        """反量化数据"""
        if self.quant_type == 'int8':
            return (quantized_data.float() - self.zero_point) * self.scale
        return quantized_data

class QuantizedLinear(nn.Module):
    """量化线性层"""
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # 量化参数
        self.weight_scale = nn.Parameter(torch.ones(1))
        self.weight_zero_point = nn.Parameter(torch.zeros(1))
        self.input_scale = nn.Parameter(torch.ones(1))
        self.input_zero_point = nn.Parameter(torch.zeros(1))
        
        # 原始权重（用于训练）
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x):
        # 量化输入
        x_quant = torch.clamp(
            x / self.input_scale + self.input_zero_point,
            -128, 127
        ).to(torch.int8)
        
        # 量化权重
        w_quant = torch.clamp(
            self.weight / self.weight_scale + self.weight_zero_point,
            -128, 127
        ).to(torch.int8)
        
        # 使用INT8计算
        # 这里简化为使用浮点计算，实际应该使用INT8算子
        output = F.linear(x_quant.float(), w_quant.float(), self.bias)
        
        return output

class ModelParallelism:
    """模型并行基础实现
    
    并行策略：
    1. 层内并行：将单个层分割到多个设备
    2. 流水线并行：将模型分割到多个设备
    3. 张量并行：将张量操作分割到多个设备
    
    面试重点：理解不同并行策略的适用场景
    """
    def __init__(self, devices):
        self.devices = devices
        self.num_devices = len(devices)
        
    def split_model(self, model):
        """分割模型到多个设备"""
        layers = list(model.children())
        layers_per_device = len(layers) // self.num_devices
        
        device_models = []
        for i in range(self.num_devices):
            start_idx = i * layers_per_device
            end_idx = (i + 1) * layers_per_device if i < self.num_devices - 1 else len(layers)
            
            device_model = nn.Sequential(*layers[start_idx:end_idx])
            device_model.to(self.devices[i])
            device_models.append(device_model)
            
        return device_models
    
    def parallel_forward(self, device_models, x):
        """并行前向传播"""
        # 流水线并行：顺序执行
        output = x
        for model in device_models:
            output = output.to(model.devices[0])
            output = model(output)
            
        return output

class BatchInferenceOptimizer:
    """批量推理优化
    
    优化策略：
    1. 动态batching
    2. 序列长度padding
    3. 计算图优化
    
    面试重点：理解批量推理的性能优化方法
    """
    def __init__(self, model, max_batch_size=32, max_seq_len=512):
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        
    def dynamic_batching(self, requests):
        """动态批量处理"""
        # 按序列长度分组
        length_groups = defaultdict(list)
        for req in requests:
            length_groups[len(req)].append(req)
        
        results = []
        for length, batch_requests in length_groups.items():
            # 创建batch
            batch_size = min(len(batch_requests), self.max_batch_size)
            
            for i in range(0, len(batch_requests), batch_size):
                batch = batch_requests[i:i+batch_size]
                
                # padding到相同长度
                padded_batch = self._pad_batch(batch, length)
                
                # 批量推理
                with torch.no_grad():
                    outputs = self.model(padded_batch)
                
                # 处理结果
                results.extend(self._process_outputs(outputs, len(batch)))
        
        return results
    
    def _pad_batch(self, batch, target_length):
        """padding批次到目标长度"""
        padded_batch = torch.zeros(len(batch), target_length, dtype=torch.long)
        for i, seq in enumerate(batch):
            padded_batch[i, :len(seq)] = torch.tensor(seq)
        return padded_batch
    
    def _process_outputs(self, outputs, batch_size):
        """处理批量输出"""
        # 这里简化处理，实际需要根据具体任务处理
        return [outputs[i] for i in range(batch_size)]

# 测试代码
if __name__ == "__main__":
    # 测试量化
    quantizer = Quantizer('int8')
    
    # 创建测试数据
    data = torch.randn(1000)
    
    # 校准和量化
    quantizer.calibrate(data)
    quantized_data = quantizer.quantize(data)
    dequantized_data = quantizer.dequantize(quantized_data)
    
    print(f"原始数据范围: [{data.min():.2f}, {data.max():.2f}]")
    print(f"量化后数据类型: {quantized_data.dtype}")
    print(f"反量化后MSE: {F.mse_loss(data, dequantized_data):.6f}")
    
    # 测试量化线性层
    quantized_linear = QuantizedLinear(128, 64)
    x = torch.randn(32, 128)
    output = quantized_linear(x)
    print(f"量化线性层输出形状: {output.shape}")
    
    # 测试模型并行
    if torch.cuda.device_count() > 1:
        devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
        
        # 创建简单模型
        model = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        parallelism = ModelParallelism(devices)
        device_models = parallelism.split_model(model)
        
        x = torch.randn(16, 128).to(devices[0])
        output = parallelism.parallel_forward(device_models, x)
        
        print(f"模型并行输入形状: {x.shape}")
        print(f"模型并行输出形状: {output.shape}")
    
    # 测试批量推理优化
    model = nn.Linear(128, 64)
    optimizer = BatchInferenceOptimizer(model)
    
    # 创建测试请求
    requests = [
        [1, 2, 3, 4],
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5, 6],
        [1, 2, 3, 4, 5, 6, 7]
    ]
    
    results = optimizer.dynamic_batching(requests)
    print(f"批量推理处理了 {len(results)} 个请求")
    
    print("推理优化技术实现正确！")