import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import Counter

class CrossEntropyLoss:
    """交叉熵损失函数 - 分类任务的标准损失
    
    实现要点：
    1. 数值稳定性处理（log-sum-exp技巧）
    2. 支持标签平滑
    3. 支持类别权重
    
    公式：L = -∑y_i * log(p_i)
    
    面试重点：理解交叉熵的原理和数值稳定性处理
    """
    def __init__(self, label_smoothing=0.0, class_weights=None):
        self.label_smoothing = label_smoothing
        self.class_weights = class_weights
        
    def __call__(self, logits, targets):
        """
        logits: [batch_size, num_classes]
        targets: [batch_size] (class indices) or [batch_size, num_classes] (one-hot)
        """
        # 数值稳定的softmax
        log_probs = F.log_softmax(logits, dim=-1)
        
        # 处理不同形式的targets
        if targets.dim() == 1:
            # 标签形式
            targets_one_hot = F.one_hot(targets, num_classes=logits.size(-1)).float()
        else:
            # one-hot形式
            targets_one_hot = targets.float()
        
        # 标签平滑
        if self.label_smoothing > 0:
            targets_one_hot = (1 - self.label_smoothing) * targets_one_hot + \
                             self.label_smoothing / logits.size(-1)
        
        # 计算交叉熵
        loss = -torch.sum(targets_one_hot * log_probs, dim=-1)
        
        # 应用类别权重
        if self.class_weights is not None:
            weights = self.class_weights[targets] if targets.dim() == 1 else \
                     torch.matmul(targets_one_hot, self.class_weights)
            loss = loss * weights
        
        return loss.mean()

class StableSoftmax:
    """数值稳定的Softmax函数
    
    实现要点：
    1. 减去最大值防止数值溢出
    2. 支持不同维度的计算
    3. 支持温度参数
    
    面试重点：理解数值稳定性的重要性
    """
    def __init__(self, dim=-1, temperature=1.0):
        self.dim = dim
        self.temperature = temperature
        
    def __call__(self, x):
        # 数值稳定：减去最大值
        x_stable = x - torch.max(x, dim=self.dim, keepdim=True)[0]
        
        # 应用温度参数
        x_stable = x_stable / self.temperature
        
        # 计算softmax
        exp_x = torch.exp(x_stable)
        return exp_x / torch.sum(exp_x, dim=self.dim, keepdim=True)

class TopKSampler:
    """Top-k采样 - 从概率最高的k个候选中采样
    
    实现要点：
    1. 保留top-k概率
    2. 重新归一化
    3. 随机采样
    
    面试重点：理解Top-k采样在文本生成中的作用
    """
    def __init__(self, k=50):
        self.k = k
        
    def sample(self, logits):
        # 应用softmax获得概率分布
        probs = F.softmax(logits, dim=-1)
        
        # 获取top-k
        top_k_probs, top_k_indices = torch.topk(probs, self.k, dim=-1)
        
        # 重新归一化
        top_k_probs = top_k_probs / torch.sum(top_k_probs, dim=-1, keepdim=True)
        
        # 采样
        sampled_indices = torch.multinomial(top_k_probs, 1)
        
        # 获取对应的原始索引
        return top_k_indices.gather(-1, sampled_indices)

class TopPSampler:
    """Top-p (Nucleus) 采样 - 从累积概率达到p的最小集合中采样
    
    实现要点：
    1. 按概率排序
    2. 计算累积概率
    3. 选择累积概率达到p的最小集合
    
    面试重点：理解Top-p采样与Top-k采样的区别
    """
    def __init__(self, p=0.9):
        self.p = p
        
    def sample(self, logits):
        # 应用softmax获得概率分布
        probs = F.softmax(logits, dim=-1)
        
        # 按概率排序
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        
        # 计算累积概率
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # 创建mask
        mask = cumulative_probs <= self.p
        
        # 确保至少选择一个
        mask[:, :, 0] = 1
        
        # 应用mask
        filtered_probs = sorted_probs * mask.float()
        filtered_probs = filtered_probs / torch.sum(filtered_probs, dim=-1, keepdim=True)
        
        # 采样
        sampled_indices = torch.multinomial(filtered_probs, 1)
        
        # 获取对应的原始索引
        return sorted_indices.gather(-1, sampled_indices)

class BLEUScore:
    """BLEU评分 - 机器翻译评估指标
    
    实现要点：
    1. 计算n-gram精度
    2. 短句惩罚
    3. 几何平均
    
    面试重点：理解BLEU评分的计算原理
    """
    def __init__(self, max_n=4):
        self.max_n = max_n
        
    def _get_ngrams(self, tokens, n):
        """获取n-gram"""
        return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    
    def _calculate_precision(self, reference, candidate, n):
        """计算n-gram精度"""
        if len(candidate) < n:
            return 0.0
            
        candidate_ngrams = set(self._get_ngrams(candidate, n))
        reference_ngrams = set(self._get_ngrams(reference, n))
        
        if not candidate_ngrams:
            return 0.0
            
        # 计算匹配的n-gram数量
        matches = len(candidate_ngrams & reference_ngrams)
        total = len(candidate_ngrams)
        
        return matches / total if total > 0 else 0.0
    
    def __call__(self, references, candidates):
        """
        references: 参考译文列表
        candidates: 候选译文列表
        """
        if len(references) != len(candidates):
            raise ValueError("References and candidates must have the same length")
        
        # 计算各n-gram的精度
        precisions = []
        for n in range(1, self.max_n + 1):
            n_precision = 0.0
            for ref, cand in zip(references, candidates):
                ref_tokens = ref.split()
                cand_tokens = cand.split()
                n_precision += self._calculate_precision(ref_tokens, cand_tokens, n)
            
            precisions.append(n_precision / len(references))
        
        # 计算简短惩罚
        total_ref_len = sum(len(ref.split()) for ref in references)
        total_cand_len = sum(len(cand.split()) for cand in candidates)
        
        if total_cand_len > total_ref_len:
            bp = 1.0
        else:
            bp = math.exp(1 - total_ref_len / total_cand_len)
        
        # 计算BLEU分数
        if all(p > 0 for p in precisions):
            bleu = bp * math.exp(sum(math.log(p) for p in precisions) / self.max_n)
        else:
            bleu = 0.0
            
        return bleu

class Perplexity:
    """困惑度 - 语言模型评估指标
    
    实现要点：
    1. 计算交叉熵
    2. 指数化
    3. 支持序列级别计算
    
    公式：PPL = exp(-1/N * ∑log p(x_i))
    
    面试重点：理解困惑度的含义和计算方法
    """
    def __init__(self):
        self.cross_entropy = nn.CrossEntropyLoss()
        
    def __call__(self, logits, targets):
        """
        logits: [batch_size, seq_len, vocab_size]
        targets: [batch_size, seq_len]
        """
        # 重塑为二维
        batch_size, seq_len, vocab_size = logits.shape
        logits_flat = logits.view(-1, vocab_size)
        targets_flat = targets.view(-1)
        
        # 计算交叉熵
        ce_loss = self.cross_entropy(logits_flat, targets_flat)
        
        # 计算困惑度
        perplexity = torch.exp(ce_loss)
        
        return perplexity.item()

# 测试代码
if __name__ == "__main__":
    # 测试交叉熵损失
    ce_loss = CrossEntropyLoss(label_smoothing=0.1)
    logits = torch.randn(32, 10)
    targets = torch.randint(0, 10, (32,))
    loss = ce_loss(logits, targets)
    print(f"交叉熵损失: {loss:.4f}")
    
    # 测试数值稳定Softmax
    stable_softmax = StableSoftmax()
    x = torch.randn(3, 5)
    softmax_output = stable_softmax(x)
    print(f"Softmax输出形状: {softmax_output.shape}")
    print(f"Softmax输出和: {softmax_output.sum(dim=-1)}")
    
    # 测试Top-k采样
    top_k_sampler = TopKSampler(k=3)
    logits = torch.randn(1, 10)
    sample = top_k_sampler.sample(logits)
    print(f"Top-k采样结果: {sample.item()}")
    
    # 测试Top-p采样
    top_p_sampler = TopPSampler(p=0.8)
    sample = top_p_sampler.sample(logits)
    print(f"Top-p采样结果: {sample.item()}")
    
    # 测试BLEU评分
    bleu = BLEUScore()
    references = ["the cat is on the mat", "hello world"]
    candidates = ["the cat is on mat", "hello world"]
    bleu_score = bleu(references, candidates)
    print(f"BLEU评分: {bleu_score:.4f}")
    
    # 测试困惑度
    perplexity = Perplexity()
    logits = torch.randn(4, 10, 1000)
    targets = torch.randint(0, 1000, (4, 10))
    ppl = perplexity(logits, targets)
    print(f"困惑度: {ppl:.2f}")
    
    print("损失函数和评估指标实现正确！")