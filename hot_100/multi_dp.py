from typing import List

def uniquePaths(m: int, n: int) -> int:
    """
    不同路径
    
    问题：从网格左上角到右下角有多少条不同的路径（只能向右或向下移动）
    
    思路：
    1. 状态定义：dp[i][j] 表示从 (0,0) 到 (i,j) 的路径数
    2. 状态转移：dp[i][j] = dp[i-1][j] + dp[i][j-1]
       - 可以从上方到达 (i,j)
       - 可以从左方到达 (i,j)
    3. 边界条件：
       - 第一行和第一列都只有 1 条路径
       - dp[0][j] = 1, dp[i][0] = 1
    
    优化：
    - 可以使用组合数学：C(m+n-2, m-1)
    - 可以优化空间复杂度到 O(n)
    
    示例：m=3, n=7
    - dp = [
      [1,1,1,1,1,1,1],
      [1,2,3,4,5,6,7],
      [1,3,6,10,15,21,28]
    ]
    - 返回 dp[2][6] = 28
    
    时间复杂度：O(mn)，空间复杂度：O(mn)
    """
    dp = [[1] * n for _ in range(m)]
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
    return dp[m - 1][n - 1]

def minPathSum(grid: List[List[int]]) -> int:
    """
    最小路径和
    
    问题：从网格左上角到右下角的最小路径和（只能向右或向下移动）
    
    思路：
    1. 状态定义：dp[i][j] 表示从 (0,0) 到 (i,j) 的最小路径和
    2. 状态转移：dp[i][j] = min(dp[i-1][j], dp[i][j-1]) + grid[i][j]
       - 选择上方和左方中路径和较小的
       - 加上当前格子的值
    3. 边界条件：
       - 第一行：只能从左方到达
       - 第一列：只能从上方到达
    
    优化：
    - 可以优化空间复杂度到 O(n)
    - 可以直接修改原数组，节省空间
    
    示例：grid = [[1,3,1],[1,5,1],[4,2,1]]
    - dp = [
      [1,4,5],
      [2,7,6],
      [6,8,7]
    ]
    - 返回 dp[2][2] = 7
    
    时间复杂度：O(mn)，空间复杂度：O(mn)
    """
    if not grid or not grid[0]:
        return 0
    m, n = len(grid), len(grid[0])
    dp = [[0] * n for _ in range(m)]
    dp[0][0] = grid[0][0]
    for i in range(1, m):
        dp[i][0] = dp[i - 1][0] + grid[i][0]
    for j in range(1, n):
        dp[0][j] = dp[0][j - 1] + grid[0][j]
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = min(dp[i - 1][j], dp[i][j - 1]) + grid[i][j]
    return dp[m - 1][n - 1]

def longestPalindromeSubseq(s: str) -> int:
    """
    最长回文子序列
    
    问题：找出字符串中最长的回文子序列长度
    
    思路：
    1. 状态定义：dp[i][j] 表示 s[i:j+1] 中最长回文子序列的长度
    2. 状态转移：
       - 如果 s[i] == s[j]：dp[i][j] = 2 + dp[i+1][j-1]
       - 如果 s[i] != s[j]：dp[i][j] = max(dp[i+1][j], dp[i][j-1])
    3. 边界条件：
       - dp[i][i] = 1（单个字符是回文）
       - dp[i][j] = 0 if i > j
    4. 计算顺序：按长度从小到大计算
    
    示例：s = "bbbab"
    - dp[0][4] = 4 ("bbbb")
    - 计算过程：
      * dp[0][0]=1, dp[1][1]=1, dp[2][2]=1, dp[3][3]=1, dp[4][4]=1
      * dp[0][1]=2, dp[1][2]=2, dp[2][3]=1, dp[3][4]=2
      * dp[0][2]=2, dp[1][3]=2, dp[2][4]=3
      * dp[0][3]=3, dp[1][4]=3
      * dp[0][4]=4
    
    时间复杂度：O(n²)，空间复杂度：O(n²)
    """
    n = len(s)
    dp = [[0] * n for _ in range(n)]
    for i in range(n):
        dp[i][i] = 1
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            if s[i] == s[j]:
                dp[i][j] = 2 + (dp[i + 1][j - 1] if length > 2 else 0)
            else:
                dp[i][j] = max(dp[i + 1][j], dp[i][j - 1])
    return dp[0][n - 1]

def longestCommonSubsequence(text1: str, text2: str) -> int:
    """
    最长公共子序列
    
    问题：找出两个字符串的最长公共子序列长度
    
    思路：
    1. 状态定义：dp[i][j] 表示 text1[0:i] 和 text2[0:j] 的最长公共子序列长度
    2. 状态转移：
       - 如果 text1[i-1] == text2[j-1]：dp[i][j] = dp[i-1][j-1] + 1
       - 如果 text1[i-1] != text2[j-1]：dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    3. 边界条件：dp[0][j] = 0, dp[i][0] = 0
    
    应用：
    - 字符串相似度计算
    - 版本控制中的差异比较
    - 生物信息学中的 DNA 序列比对
    
    示例：text1 = "abcde", text2 = "ace"
    - dp[5][3] = 3 ("ace")
    - 计算过程：
      * text1[0] = 'a', text2[0] = 'a' -> dp[1][1] = 1
      * text1[1] = 'b', text2[1] = 'c' -> dp[2][2] = 1
      * text1[2] = 'c', text2[1] = 'c' -> dp[3][2] = 2
      * text1[4] = 'e', text2[2] = 'e' -> dp[5][3] = 3
    
    时间复杂度：O(mn)，空间复杂度：O(mn)
    """
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]

def minDistance(word1: str, word2: str) -> int:
    """
    编辑距离（Levenshtein距离）动态规划解法
    
    问题：计算将 word1 转换为 word2 所需的最少编辑操作次数
    编辑操作包括：插入、删除、替换字符
    
    思路：
    1. 状态定义：dp[i][j] 表示 word1[0:i] 转换为 word2[0:j] 的最小编辑距离
    2. 状态转移方程：
       - 当 word1[i-1] == word2[j-1] 时：
         dp[i][j] = dp[i-1][j-1]  （字符相同，不需要操作）
         # 解释：如果当前字符相同，说明不需要额外操作，最短编辑距离和前面子问题一致。
         # 这是因为编辑距离的本质是将 word1 的前 i 个字符变成 word2 的前 j 个字符，
         # 如果最后一个字符已经相等，只需考虑前 i-1 和 j-1 的最优解。
       - 当 word1[i-1] != word2[j-1] 时：
         dp[i][j] = min(
             dp[i-1][j] + 1,     # 删除 word1[i-1]
             dp[i][j-1] + 1,     # 插入 word2[j-1]
             dp[i-1][j-1] + 1    # 替换 word1[i-1] 为 word2[j-1]
         )
         # 解释：如果当前字符不同，最短编辑距离一定是三种操作中最优的那一个：
         # 1. 删除 word1[i-1]，相当于把 word1[0:i-1] 变成 word2[0:j]，再加一次删除操作；
         # 2. 插入 word2[j-1]，相当于把 word1[0:i] 变成 word2[0:j-1]，再加一次插入操作；
         # 3. 替换 word1[i-1] 为 word2[j-1]，相当于把 word1[0:i-1] 变成 word2[0:j-1]，再加一次替换操作。
         # 由于每次只允许一步操作，递归地选择三者中最优的路径，最终就能得到最短编辑距离。
         # 这种定义方式保证了每个子问题的最优性，符合动态规划的最优子结构性质，因此能够正确求解最短编辑距离。
         )
    3. 边界条件：
       - dp[i][0] = i  （word1[0:i] 转换为空串需要 i 次删除）
       - dp[0][j] = j  （空串转换为 word2[0:j] 需要 j 次插入）
    
    应用：
    - 拼写检查和纠正
    - DNA 序列比对
    - 自然语言处理中的相似度计算
    
    示例：word1 = "horse", word2 = "ros"
    - dp[5][3] = 3
    - 操作序列：horse -> rorse (替换 h->r) -> rose (删除 r) -> ros (删除 e)
    
    时间复杂度：O(mn)，空间复杂度：O(mn)
    """
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i - 1] == word2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
    return dp[m][n]