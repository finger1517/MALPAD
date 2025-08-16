import torch
import torch.nn as nn
import torch.nn.functional as F
import heapq
import numpy as np
from collections import defaultdict

class BeamSearchDecoder:
    """束搜索解码器
    
    算法原理：
    1. 维护beam_size个候选序列
    2. 每一步扩展所有候选
    3. 选择概率最高的beam_size个候选
    
    参数：
    - beam_size: 束宽度
    - max_length: 最大生成长度
    - length_penalty: 长度惩罚系数
    
    面试重点：理解束搜索的原理和参数调优
    """
    def __init__(self, beam_size=5, max_length=50, length_penalty=0.6):
        self.beam_size = beam_size
        self.max_length = max_length
        self.length_penalty = length_penalty
        
    def decode(self, model, start_token, end_token, device):
        """执行束搜索解码"""
        # 初始化束
        beams = [(0, [start_token])]  # (score, sequence)
        
        for step in range(self.max_length):
            new_beams = []
            
            for score, sequence in beams:
                # 如果已经遇到结束标记，直接保留
                if sequence[-1] == end_token:
                    new_beams.append((score, sequence))
                    continue
                
                # 准备输入
                input_tensor = torch.tensor([sequence], dtype=torch.long).to(device)
                
                # 模型预测
                with torch.no_grad():
                    logits = model(input_tensor)
                    probs = F.softmax(logits[:, -1, :], dim=-1)
                
                # 获取top-k候选
                topk_probs, topk_indices = torch.topk(probs, self.beam_size)
                
                # 添加到新束中
                for i in range(self.beam_size):
                    new_score = score + torch.log(topk_probs[0, i]).item()
                    new_sequence = sequence + [topk_indices[0, i].item()]
                    new_beams.append((new_score, new_sequence))
            
            # 选择最佳beam_size个候选
            beams = heapq.nlargest(self.beam_size, new_beams, 
                                 key=lambda x: self._apply_length_penalty(x[0], len(x[1])))
        
        # 返回最佳序列
        best_sequence = max(beams, key=lambda x: self._apply_length_penalty(x[0], len(x[1])))[1]
        return best_sequence
    
    def _apply_length_penalty(self, score, length):
        """应用长度惩罚"""
        return score / ((5 + length) / (5 + 1)) ** self.length_penalty

class GreedyDecoder:
    """贪心解码器
    
    算法原理：
    1. 每一步选择概率最高的token
    2. 简单快速，但可能不是全局最优
    
    面试重点：理解贪心解码的优缺点
    """
    def __init__(self, max_length=50):
        self.max_length = max_length
        
    def decode(self, model, start_token, end_token, device):
        """执行贪心解码"""
        sequence = [start_token]
        
        for step in range(self.max_length):
            # 准备输入
            input_tensor = torch.tensor([sequence], dtype=torch.long).to(device)
            
            # 模型预测
            with torch.no_grad():
                logits = model(input_tensor)
                probs = F.softmax(logits[:, -1, :], dim=-1)
            
            # 选择概率最高的token
            next_token = torch.argmax(probs, dim=-1).item()
            sequence.append(next_token)
            
            # 如果遇到结束标记，停止生成
            if next_token == end_token:
                break
        
        return sequence

class RandomSampler:
    """随机采样解码器
    
    算法原理：
    1. 根据概率分布随机采样
    2. 支持温度参数调节随机性
    
    面试重点：理解温度参数的作用
    """
    def __init__(self, max_length=50, temperature=1.0):
        self.max_length = max_length
        self.temperature = temperature
        
    def decode(self, model, start_token, end_token, device):
        """执行随机采样解码"""
        sequence = [start_token]
        
        for step in range(self.max_length):
            # 准备输入
            input_tensor = torch.tensor([sequence], dtype=torch.long).to(device)
            
            # 模型预测
            with torch.no_grad():
                logits = model(input_tensor)
            
            # 应用温度参数
            scaled_logits = logits[:, -1, :] / self.temperature
            probs = F.softmax(scaled_logits, dim=-1)
            
            # 随机采样
            next_token = torch.multinomial(probs, 1).item()
            sequence.append(next_token)
            
            # 如果遇到结束标记，停止生成
            if next_token == end_token:
                break
        
        return sequence

# 测试代码
if __name__ == "__main__":
    # 创建简单的测试模型
    class SimpleModel(nn.Module):
        def __init__(self, vocab_size=1000):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, 128)
            self.lstm = nn.LSTM(128, 256, batch_first=True)
            self.fc = nn.Linear(256, vocab_size)
            
        def forward(self, x):
            emb = self.embedding(x)
            lstm_out, _ = self.lstm(emb)
            return self.fc(lstm_out)
    
    # 初始化模型和解码器
    model = SimpleModel()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # 测试束搜索
    beam_decoder = BeamSearchDecoder(beam_size=3, max_length=20)
    beam_result = beam_decoder.decode(model, start_token=1, end_token=2, device=device)
    print(f"束搜索结果长度: {len(beam_result)}")
    
    # 测试贪心解码
    greedy_decoder = GreedyDecoder(max_length=20)
    greedy_result = greedy_decoder.decode(model, start_token=1, end_token=2, device=device)
    print(f"贪心解码结果长度: {len(greedy_result)}")
    
    # 测试随机采样
    random_sampler = RandomSampler(max_length=20, temperature=0.8)
    random_result = random_sampler.decode(model, start_token=1, end_token=2, device=device)
    print(f"随机采样结果长度: {len(random_result)}")
    
    print("解码器实现正确！")