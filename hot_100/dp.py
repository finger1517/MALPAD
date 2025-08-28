from typing import List

def climbStairs(n: int) -> int:
    """
    爬楼梯
    
    问题：爬 n 阶楼梯，每次可以爬 1 或 2 阶，有多少种不同的方法
    
    思路：
    1. 状态定义：dp[i] 表示爬到第 i 阶楼梯的方法数
    2. 状态转移：dp[i] = dp[i-1] + dp[i-2]
       - 从第 i-1 阶爬 1 阶
       - 从第 i-2 阶爬 2 阶
    3. 边界条件：dp[1] = 1, dp[2] = 2
    4. 优化：可以只用两个变量代替整个数组
    
    示例：n = 3
    - dp[1] = 1, dp[2] = 2
    - dp[3] = dp[2] + dp[1] = 3
    
    时间复杂度：O(n)，空间复杂度：O(n)（可优化到 O(1)）
    """
    if n <= 2:
        return n
    dp = [0] * (n + 1)
    dp[1] = 1
    dp[2] = 2
    for i in range(3, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]
    return dp[n]

def generate(numRows: int) -> List[List[int]]:
    """
    杨辉三角
    
    问题：生成前 numRows 行的杨辉三角
    
    思路：
    1. 杨辉三角的性质：
       - 每行的第一个和最后一个数都是 1
       - 中间的数等于上一行相邻两个数的和
    2. 逐行生成，利用上一行的结果
    
    示例：numRows = 5
    [
        [1],
        [1,1],
        [1,2,1],
        [1,3,3,1],
        [1,4,6,4,1]
    ]
    
    时间复杂度：O(n²)，空间复杂度：O(n²)
    """
    if numRows == 0:
        return []
    triangle = [[1]]
    for i in range(1, numRows):
        prev_row = triangle[-1]
        new_row = [1]
        for j in range(1, i):
            new_row.append(prev_row[j - 1] + prev_row[j])
        new_row.append(1)
        triangle.append(new_row)
    return triangle

def rob(nums: List[int]) -> int:
    """
    打家劫舍
    
    问题：不能抢劫相邻的房屋，求能抢劫的最大金额
    
    思路：
    1. 状态定义：dp[i] 表示抢劫到第 i 个房屋时的最大金额
    2. 状态转移：
       - 如果抢劫第 i 个房屋：dp[i] = dp[i-2] + nums[i]
       - 如果不抢劫第 i 个房屋：dp[i] = dp[i-1]
       - 取两者的最大值
    3. 边界条件：dp[0] = nums[0], dp[1] = max(nums[0], nums[1])
    
    示例：nums = [2,7,9,3,1]
    - dp[0] = 2
    - dp[1] = max(2, 7) = 7
    - dp[2] = max(7, 2+9) = 11
    - dp[3] = max(11, 7+3) = 11
    - dp[4] = max(11, 11+1) = 12
    
    时间复杂度：O(n)，空间复杂度：O(n)（可优化到 O(1)）
    """
    if not nums:
        return 0
    if len(nums) == 1:
        return nums[0]
    dp = [0] * len(nums)
    dp[0] = nums[0]
    dp[1] = max(nums[0], nums[1])
    for i in range(2, len(nums)):
        dp[i] = max(dp[i - 1], dp[i - 2] + nums[i])
    return dp[-1]

def numSquares(n: int) -> int:
    """
    完全平方数
    
    问题：将 n 分解为完全平方数的和，求最少需要多少个完全平方数
    
    思路：
    1. 状态定义：dp[i] 表示组成 i 的最少完全平方数个数
    2. 状态转移：dp[i] = min(dp[i], dp[i - j*j] + 1)
       - 遍历所有小于等于 i 的完全平方数 j*j
       - 选择使用这个完全平方数后的最优解
    3. 边界条件：dp[0] = 0
    
    示例：n = 12
    - dp[12] = min(dp[12-1*1]+1, dp[12-2*2]+1, dp[12-3*3]+1)
    - = min(dp[11]+1, dp[8]+1, dp[3]+1)
    - = min(3+1, 2+1, 3+1) = 3
    
    时间复杂度：O(n√n)，空间复杂度：O(n)
    """
    dp = [float('inf')] * (n + 1)
    dp[0] = 0
    for i in range(1, n + 1):
        j = 1
        while j * j <= i:
            dp[i] = min(dp[i], dp[i - j * j] + 1)
            j += 1
    return dp[n]

def coinChange(coins: List[int], amount: int) -> int:
    """
    零钱兑换
    
    问题：用最少的硬币凑出目标金额，每种硬币有无限个
    
    思路：
    1. 状态定义：dp[i] 表示凑出金额 i 所需的最少硬币数
    2. 状态转移：dp[i] = min(dp[i], dp[i-coin] + 1)
       - 遍历所有硬币面值
       - 如果使用该硬币，数量为 dp[i-coin] + 1
    3. 边界条件：dp[0] = 0
    
    示例：coins = [1,2,5], amount = 11
    - dp[11] = min(dp[10]+1, dp[9]+1, dp[6]+1)
    - = min(3+1, 4+1, 2+1) = 3
    
    时间复杂度：O(amount * len(coins))，空间复杂度：O(amount)
    """
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    for i in range(1, amount + 1):
        for coin in coins:
            if i >= coin:
                dp[i] = min(dp[i], dp[i - coin] + 1)
    return dp[amount] if dp[amount] != float('inf') else -1

def wordBreak(s: str, wordDict: List[str]) -> bool:
    """
    单词拆分
    
    问题：判断字符串是否可以由字典中的单词组成
    
    思路：
    1. 状态定义：dp[i] 表示 s[0:i] 是否可以由字典中的单词组成
    2. 状态转移：如果存在 j 使得 dp[j] 为 True 且 s[j:i] 在字典中，则 dp[i] = True
    3. 边界条件：dp[0] = True（空字符串可以组成）
    
    示例：s = "leetcode", wordDict = ["leet", "code"]
    - dp[0] = True
    - dp[4] = True ("leet" 在字典中)
    - dp[8] = True (s[4:8] = "code" 在字典中)
    
    时间复杂度：O(n²)，空间复杂度：O(n)
    """
    word_set = set(wordDict)
    n = len(s)
    dp = [False] * (n + 1)
    dp[0] = True
    for i in range(1, n + 1):
        for j in range(i):
            if dp[j] and s[j:i] in word_set:
                dp[i] = True
                break
    return dp[n]

def lengthOfLIS(nums: List[int]) -> int:
    """
    最长递增子序列
    
    问题：找出数组中最长的严格递增子序列的长度
    
    思路：
    1. 状态定义：dp[i] 表示以 nums[i] 结尾的最长递增子序列长度
    2. 状态转移：对于每个 i，遍历前面的所有 j
       - 如果 nums[i] > nums[j]，则 dp[i] = max(dp[i], dp[j] + 1)
    3. 边界条件：dp[i] = 1（每个元素本身就是长度为 1 的子序列）
    
    示例：nums = [10,9,2,5,3,7,101,18]
    - dp[0] = 1
    - dp[1] = 1 (9 < 10)
    - dp[2] = 1 (2 < 10, 2 < 9)
    - dp[3] = max(1, dp[2]+1) = 2 (5 > 2)
    - dp[4] = max(1, dp[2]+1) = 2 (3 > 2)
    - dp[5] = max(1, dp[3]+1, dp[4]+1) = 3 (7 > 5, 7 > 3)
    
    时间复杂度：O(n²)，空间复杂度：O(n)
    """
    if not nums:
        return 0
    dp = [1] * len(nums)
    for i in range(1, len(nums)):
        for j in range(i):
            if nums[i] > nums[j]:
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp)

def maxProduct(nums: List[int]) -> int:
    """
    乘积最大子数组
    两个dp数组，一个记录最大值，一个记录最小值
    
    问题：找出数组中乘积最大的连续子数组
    
    思路：
    1. 需要维护两个数组：max_dp 和 min_dp
       - max_dp[i] 表示以 nums[i] 结尾的最大乘积
       - min_dp[i] 表示以 nums[i] 结尾的最小乘积（用于处理负数）
    2. 状态转移：
       - 如果 nums[i] > 0：
         * max_dp[i] = max(nums[i], max_dp[i-1] * nums[i])
         * min_dp[i] = min(nums[i], min_dp[i-1] * nums[i])
       - 如果 nums[i] < 0：
         * max_dp[i] = max(nums[i], min_dp[i-1] * nums[i])
         * min_dp[i] = min(nums[i], max_dp[i-1] * nums[i])
    
    示例：nums = [2,3,-2,4]
    - max_dp[0] = 2, min_dp[0] = 2
    - max_dp[1] = 6, min_dp[1] = 3
    - max_dp[2] = -2, min_dp[2] = -12
    - max_dp[3] = 4, min_dp[3] = -48
    - result = max(2, 6, -2, 4) = 6
    
    时间复杂度：O(n)，空间复杂度：O(n)（可优化到 O(1)）
    """
    if not nums:
        return 0
    max_dp = [0] * len(nums)
    min_dp = [0] * len(nums)
    max_dp[0] = min_dp[0] = nums[0]
    result = nums[0]
    for i in range(1, len(nums)):
        if nums[i] > 0:
            max_dp[i] = max(nums[i], max_dp[i - 1] * nums[i])
            min_dp[i] = min(nums[i], min_dp[i - 1] * nums[i])
        else:
            max_dp[i] = max(nums[i], min_dp[i - 1] * nums[i])
            min_dp[i] = min(nums[i], max_dp[i - 1] * nums[i])
        result = max(result, max_dp[i])
    return result

def canPartition(nums: List[int]) -> bool:
    """
    分割等和子集

    
    问题：判断数组是否可以分割成两个子集，使得两个子集的和相等
    
    思路：
    1. 首先计算总和，如果总和是奇数，直接返回 False
    2. 目标是找到子集和为 total // 2
    3. 转化为 0-1 背包问题：
       - dp[i] 表示是否可以组成和为 i
       - 对于每个 num，从后往前更新 dp 数组
    4. 状态转移：dp[i] = dp[i] or dp[i-num]
    
    示例：nums = [1,5,11,5]
    - total = 22, target = 11
    - dp[0] = True
    - num=1: dp[1] = True
    - num=5: dp[6] = True
    - num=11: dp[11] = True
    
    时间复杂度：O(n * target)，空间复杂度：O(target)
    """
    total = sum(nums)
    if total % 2 != 0:
        return False
    target = total // 2
    dp = [False] * (target + 1)
    dp[0] = True
    for num in nums:
        for i in range(target, num - 1, -1):
            dp[i] = dp[i] or dp[i - num]
    return dp[target]

def longestValidParentheses(s: str) -> int:
    """
    最长有效括号
    
    问题：找出最长有效（格式正确且连续）括号子串的长度
    
    思路：
    1. 状态定义：dp[i] 表示以 s[i] 结尾的最长有效括号子串长度
    2. 状态转移：
       - 如果 s[i] == ')'：
         * 如果 s[i-1] == '('，则 dp[i] = dp[i-2] + 2
         * 如果 s[i-1] == ')'，需要检查 s[i-dp[i-1]-1] 是否为 '('
           - 如果是，则 dp[i] = dp[i-1] + dp[i-dp[i-1]-2] + 2
    3. 注意边界条件处理
    
    示例：s = ")()())"
    - dp[0] = 0
    - dp[1] = 0
    - dp[2] = 2 (s[1]='(', s[2]=')')
    - dp[3] = 0
    - dp[4] = 2 (s[3]='(', s[4]=')')
    - dp[5] = 4 (s[0]='(', s[5]=')', dp[4]=2)
    
    时间复杂度：O(n)，空间复杂度：O(n)
    """
    if not s:
        return 0
    dp = [0] * len(s)
    max_len = 0
    for i in range(1, len(s)):
        if s[i] == ')':
            if s[i - 1] == '(':
                dp[i] = (dp[i - 2] if i >= 2 else 0) + 2
            elif i - dp[i - 1] > 0 and s[i - dp[i - 1] - 1] == '(':
                dp[i] = dp[i - 1] + (dp[i - dp[i - 1] - 2] if i - dp[i - 1] >= 2 else 0) + 2
            max_len = max(max_len, dp[i])
    return max_len