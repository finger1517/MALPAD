from typing import List

def maxProfit(prices: List[int]) -> int:
    """
    买卖股票的最佳时机
    
    问题：在股票价格数组中，选择一天买入，另一天卖出，获得最大利润
    
    思路：
    1. 遍历价格数组，维护最小价格和最大利润
    2. 对于每个价格，更新最小价格
    3. 计算当前价格与最小价格的差值，更新最大利润
    
    贪心策略：
    - 不断更新最小价格，确保买入价格最低
    - 计算每个卖出点的利润，取最大值
    
    示例：prices = [7,1,5,3,6,4]
    - min_price = 7, max_profit = 0
    - price=1: min_price=1, max_profit=0
    - price=5: min_price=1, max_profit=4
    - price=3: min_price=1, max_profit=4
    - price=6: min_price=1, max_profit=5
    - price=4: min_price=1, max_profit=5
    
    时间复杂度：O(n)，空间复杂度：O(1)
    """
    if not prices:
        return 0
    min_price = prices[0]
    max_profit = 0
    for price in prices:
        min_price = min(min_price, price)
        max_profit = max(max_profit, price - min_price)
    return max_profit

def canJump(nums: List[int]) -> bool:
    """
    跳跃游戏
    
    问题：判断是否能从数组的第一个位置跳到最后一个位置
    
    思路：
    1. 维护能够到达的最远位置 max_reach
    2. 遍历数组，对于每个位置 i：
       - 如果 i > max_reach，说明无法到达当前位置，返回 False
       - 更新 max_reach = max(max_reach, i + nums[i])
    3. 如果能遍历完整个数组，返回 True
    
    贪心策略：
    - 在每个位置，选择能够到达的最远位置
    - 只要能到达当前位置，就继续向前探索
    
    示例：nums = [2,3,1,1,4]
    - max_reach = 0
    - i=0: max_reach = max(0, 0+2) = 2
    - i=1: max_reach = max(2, 1+3) = 4
    - i=2: max_reach = max(4, 2+1) = 4
    - i=3: max_reach = max(4, 3+1) = 4
    - i=4: max_reach = max(4, 4+4) = 8
    - 返回 True
    
    时间复杂度：O(n)，空间复杂度：O(1)
    """
    max_reach = 0
    for i in range(len(nums)):
        if i > max_reach:
            return False
        max_reach = max(max_reach, i + nums[i])
    return True

def jump(nums: List[int]) -> int:
    """
    跳跃游戏 II
    
    问题：从数组第一个位置跳到最后一个位置，求最小跳跃次数
    
    思路：
    1. 维护三个变量：
       - jumps: 跳跃次数
       - current_end: 当前跳跃能到达的最远位置
       - farthest: 在当前跳跃范围内能到达的最远位置
    2. 遍历数组，对于每个位置：
       - 更新 farthest = max(farthest, i + nums[i])
       - 如果到达当前跳跃的边界，需要增加一次跳跃
    
    贪心策略：
    - 在每个跳跃范围内，选择能到达最远的位置
    - 只有在需要时才进行跳跃
    
    示例：nums = [2,3,1,1,4]
    - jumps=0, current_end=0, farthest=0
    - i=0: farthest=2, i==current_end -> jumps=1, current_end=2
    - i=1: farthest=4
    - i=2: farthest=4, i==current_end -> jumps=2, current_end=4
    - 返回 2
    
    时间复杂度：O(n)，空间复杂度：O(1)
    """
    jumps = 0
    current_end = 0
    farthest = 0
    for i in range(len(nums) - 1):
        farthest = max(farthest, i + nums[i])
        if i == current_end:
            jumps += 1
            current_end = farthest
    return jumps

def partitionLabels(s: str) -> List[int]:
    """
    划分字母区间
    
    问题：将字符串分割成尽可能多的片段，同一字母最多出现在一个片段中
    
    思路：
    1. 首先记录每个字符最后出现的位置
    2. 遍历字符串，维护当前片段的结束位置
    3. 对于每个字符，更新结束位置为该字符最后出现的位置
    4. 当到达当前片段的结束位置时，记录片段长度
    
    贪心策略：
    - 每个片段尽可能小，但要包含该片段中所有字符的最后出现位置
    - 扩展片段直到满足条件
    
    示例：s = "ababcbacadefegdehijhklij"
    - 最后位置：a=8, b=5, c=7, d=14, e=15, f=11, g=13, h=19, i=22, j=23, k=20, l=21
    - 片段：
      * "ababcbaca": 长度 9
      * "defegde": 长度 7
      * "hijhklij": 长度 8
    
    时间复杂度：O(n)，空间复杂度：O(1)（只有26个字母）
    """
    last_pos = {}
    for i, char in enumerate(s):
        last_pos[char] = i
    start = 0
    end = 0
    result = []
    for i, char in enumerate(s):
        end = max(end, last_pos[char])
        if i == end:
            result.append(end - start + 1)
            start = i + 1
    return result