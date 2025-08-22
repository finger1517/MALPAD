from typing import List, Dict

def twoSum(nums: List[int], target: int) -> List[int]:
    """
    两数之和
    
    问题：在数组中找出两个数，使它们的和等于目标值
    
    思路：
    1. 使用哈希表（字典）存储已经遍历过的数字及其索引
    2. 对于每个数字 n，检查 target - n 是否在哈希表中
    3. 如果存在，返回两个索引
    4. 如果不存在，将当前数字和索引加入哈希表
    
    哈希表优势：
    - 查找时间复杂度为 O(1)
    - 避免了暴力解法的 O(n²) 时间复杂度
    
    示例：nums = [2,7,11,15], target = 9
    - i=0, n=2: 9-2=7 不在哈希表中，存储 {2:0}
    - i=1, n=7: 9-7=2 在哈希表中，返回 [0,1]
    
    时间复杂度：O(n)，空间复杂度：O(n)
    """
    d: Dict[int, int] = {}
    for i, n in enumerate(nums):
        if target - n in d:
            return [d[target - n], i]
        d[n] = i

def groupAnagrams(strs: List[str]) -> List[List[str]]:
    """
    字母异位词分组
    
    问题：将字母异位词（字母相同但顺序不同的单词）分组
    
    思路：
    1. 对于每个字符串，将其排序作为哈希表的键
    2. 异位词排序后结果相同，会被分到同一组
    3. 使用字典存储分组结果
    
    优化：
    - 可以使用字符计数作为键，避免排序
    - 排序时间复杂度 O(k log k)，k 为字符串长度
    
    示例：strs = ["eat","tea","tan","ate","nat","bat"]
    - "eat" -> sorted = ('a','e','t')
    - "tea" -> sorted = ('a','e','t')
    - "tan" -> sorted = ('a','n','t')
    - 分组结果：[["eat","tea","ate"],["tan","nat"],["bat"]]
    
    时间复杂度：O(nk log k)，空间复杂度：O(nk)
    """
    res: Dict[tuple, List[str]] = {}
    for s in strs:
        key = tuple(sorted(s))
        res.setdefault(key, []).append(s)
    return list(res.values())


def longestConsecutive(nums: List[int]) -> int:
    """
    最长连续序列
    
    问题：找出数组中未排序的连续序列的最长长度
    
    思路：
    1. 首先将数组转换为集合，实现 O(1) 查找
    2. 遍历集合中的每个数字：
       - 只有当 n-1 不在集合中时，才开始计数（避免重复计算）
       - 从 n 开始，不断查找 n+1 是否在集合中
       - 记录连续序列的长度
    3. 返回最大长度
    
    优化：
    - 使用集合避免重复计算
    - 只从序列的起始点开始计算
    
    示例：nums = [100,4,200,1,3,2]
    - 集合：{100,4,200,1,3,2}
    - 100: 99不在集合中，序列 [100] -> 长度1
    - 4: 3在集合中，跳过
    - 200: 199不在集合中，序列 [200] -> 长度1
    - 1: 0不在集合中，序列 [1,2,3,4] -> 长度4
    - 返回 4
    
    时间复杂度：O(n)，空间复杂度：O(n)
    """
    num_set = set(nums)
    max_len = 0
    for n in num_set:
        if n - 1 not in num_set:
            cur = n
            cnt = 1
            while cur + 1 in num_set:
                cur += 1
                cnt += 1
            max_len = max(max_len, cnt)
    return max_len

