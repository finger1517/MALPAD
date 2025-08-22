from typing import List, Optional

def numIslands(grid: List[List[str]]) -> int:
    """
    岛屿数量
    
    问题：计算二维网格中岛屿的数量（'1'表示陆地，'0'表示水）
    
    思路：
    1. 使用深度优先搜索（DFS）遍历每个陆地
    2. 遇到 '1' 时，将其标记为已访问（改为 '0'），并递归访问其四个方向
    3. 每次启动 DFS 表示发现一个新岛屿
    4. 也可以使用广度优先搜索（BFS）
    
    示例：grid = [
        ["1","1","1","1","0"],
        ["1","1","0","1","0"],
        ["1","1","0","0","0"],
        ["0","0","0","0","0"]
    ]
    - 发现 (0,0) 是陆地，DFS 标记所有相连的陆地
    - 继续搜索，发现 (2,0) 是陆地，DFS 标记
    - 总共 1 个岛屿
    
    时间复杂度：O(mn)，空间复杂度：O(mn)（递归栈）
    """
    if not grid:
        return 0
    m, n = len(grid), len(grid[0])
    count = 0
    def dfs(i: int, j: int) -> None:
        if i < 0 or i >= m or j < 0 or j >= n or grid[i][j] != '1':
            return
        grid[i][j] = '0'
        dfs(i + 1, j)
        dfs(i - 1, j)
        dfs(i, j + 1)
        dfs(i, j - 1)
    for i in range(m):
        for j in range(n):
            if grid[i][j] == '1':
                count += 1
                dfs(i, j)
    return count

def orangesRotting(grid: List[List[int]]) -> int:
    """
    腐烂的橘子
    
    问题：计算所有新鲜橘子变腐烂所需的最小分钟数，如果不可能则返回 -1
    
    思路：
    1. 使用广度优先搜索（BFS）模拟腐烂过程
    2. 首先统计新鲜橘子数量，并将所有腐烂橘子加入队列
    3. 每分钟处理一层腐烂橘子，感染相邻的新鲜橘子
    4. 最后检查是否还有新鲜橘子剩余
    
    示例：grid = [[2,1,1],[1,1,0],[0,1,1]]
    - 初始：腐烂橘子在 (0,0)，新鲜橘子有 5 个
    - 第1分钟：(0,1), (1,0) 腐烂
    - 第2分钟：(0,2), (1,1), (2,1) 腐烂
    - 第3分钟：(2,2) 腐烂
    - 所有橘子都腐烂，返回 3
    
    时间复杂度：O(mn)，空间复杂度：O(mn)
    """
    if not grid:
        return -1
    m, n = len(grid), len(grid[0])
    fresh = 0
    queue = []
    for i in range(m):
        for j in range(n):
            if grid[i][j] == 2:
                queue.append((i, j))
            elif grid[i][j] == 1:
                fresh += 1
    if fresh == 0:
        return 0
    minutes = 0
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    while queue:
        for _ in range(len(queue)):
            i, j = queue.pop(0)
            for di, dj in directions:
                ni, nj = i + di, j + dj
                if 0 <= ni < m and 0 <= nj < n and grid[ni][nj] == 1:
                    grid[ni][nj] = 2
                    fresh -= 1
                    queue.append((ni, nj))
        if queue:
            minutes += 1
    return minutes if fresh == 0 else -1

def canFinish(numCourses: int, prerequisites: List[List[int]]) -> bool:
    """
    课程表
    
    问题：判断是否能完成所有课程（课程有依赖关系）
    
    思路：
    1. 将问题转化为有向图，判断是否存在环
    2. 使用深度优先搜索（DFS）检测环
    3. visited 数组状态：
       - 0: 未访问
       - 1: 正在访问（当前路径上）
       - 2: 已访问完成
    4. 如果遇到状态为 1 的节点，说明存在环
    
    示例：numCourses = 2, prerequisites = [[1,0]]
    - 课程 1 依赖于课程 0
    - 可以完成课程 0 -> 课程 1
    - 返回 True
    
    示例：numCourses = 2, prerequisites = [[1,0],[0,1]]
    - 课程 1 依赖于课程 0，课程 0 依赖于课程 1
    - 存在环，无法完成
    - 返回 False
    
    时间复杂度：O(V+E)，空间复杂度：O(V+E)
    """
    graph = [[] for _ in range(numCourses)]
    visited = [0] * numCourses
    for course, prereq in prerequisites:
        graph[course].append(prereq)
    def hasCycle(course: int) -> bool:
        if visited[course] == 1:
            return True
        if visited[course] == 2:
            return False
        visited[course] = 1
        for prereq in graph[course]:
            if hasCycle(prereq):
                return True
        visited[course] = 2
        return False
    for course in range(numCourses):
        if hasCycle(course):
            return False
    return True

class TrieNode:
    """
    字典树节点
    
    属性：
    - children: 子节点字典，键为字符，值为 TrieNode
    - is_end: 标记是否是单词的结尾
    """
    def __init__(self):
        self.children = {}
        self.is_end = False

class Trie:
    """
    字典树（前缀树）
    
    功能：
    - 插入单词
    - 搜索单词
    - 搜索前缀
    
    应用：
    - 自动补全
    - 拼写检查
    - IP路由
    """
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word: str) -> None:
        """
        插入单词到字典树
        
        思路：
        1. 从根节点开始，逐个字符向下遍历
        2. 如果字符不存在，创建新节点
        3. 最后一个字符标记为单词结尾
        
        时间复杂度：O(m)，其中 m 是单词长度
        """
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True
    
    def search(self, word: str) -> bool:
        """
        搜索单词是否在字典树中
        
        思路：
        1. 从根节点开始，逐个字符向下遍历
        2. 如果字符不存在，返回 False
        3. 最后检查是否是单词结尾
        
        时间复杂度：O(m)
        """
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end
    
    def startsWith(self, prefix: str) -> bool:
        """
        搜索前缀是否存在
        
        思路：
        1. 从根节点开始，逐个字符向下遍历
        2. 如果字符不存在，返回 False
        3. 所有字符都存在，返回 True
        
        时间复杂度：O(m)
        """
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True

class WordDictionary:
    """
    添加与搜索单词 - 数据结构设计
    
    功能：
    - 添加单词
    - 搜索单词（支持 '.' 通配符）
    
    应用：
    - 模糊搜索
    - 正则表达式匹配
    """
    def __init__(self):
        self.root = TrieNode()
    
    def addWord(self, word: str) -> None:
        """
        添加单词到字典树
        
        时间复杂度：O(m)
        """
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True
    
    def search(self, word: str) -> bool:
        """
        搜索单词（支持 '.' 通配符）
        
        思路：
        1. 使用深度优先搜索（DFS）处理通配符
        2. 遇到 '.' 时，尝试所有可能的子节点
        3. 遇到普通字符时，按正常路径搜索
        
        时间复杂度：O(26^m)（最坏情况，全是通配符）
        """
        def dfs(node: TrieNode, index: int) -> bool:
            if index == len(word):
                return node.is_end
            char = word[index]
            if char == '.':
                for child in node.children.values():
                    if dfs(child, index + 1):
                        return True
                return False
            else:
                if char not in node.children:
                    return False
                return dfs(node.children[char], index + 1)
        return dfs(self.root, 0)

def exist(board: List[List[str]], word: str) -> bool:
    """
    单词搜索
    
    问题：在二维网格中搜索单词是否存在
    
    思路：
    1. 使用深度优先搜索（DFS）从每个可能的起始位置开始搜索
    2. 搜索过程中标记已访问的字符，避免重复使用
    3. 回溯时恢复原始字符
    4. 四个方向：上、下、左、右
    
    示例：board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCCED"
    - 从 (0,0) 开始，A -> B -> C -> C -> E -> D
    - 找到单词，返回 True
    
    时间复杂度：O(mn * 4^L)，其中 L 是单词长度
    空间复杂度：O(L)（递归栈）
    """
    if not board or not board[0]:
        return False
    m, n = len(board), len(board[0])
    def dfs(i: int, j: int, index: int) -> bool:
        if index == len(word):
            return True
        if i < 0 or i >= m or j < 0 or j >= n or board[i][j] != word[index]:
            return False
        temp = board[i][j]
        board[i][j] = '#'
        found = (dfs(i + 1, j, index + 1) or
                dfs(i - 1, j, index + 1) or
                dfs(i, j + 1, index + 1) or
                dfs(i, j - 1, index + 1))
        board[i][j] = temp
        return found
    for i in range(m):
        for j in range(n):
            if dfs(i, j, 0):
                return True
    return False

def findWords(board: List[List[str]], words: List[str]) -> List[str]:
    """
    单词搜索 II
    
    问题：在二维网格中找出所有字典中的单词
    
    思路：
    1. 首先构建字典树，将所有单词插入
    2. 使用深度优先搜索（DFS）遍历网格
    3. 在搜索过程中，如果当前路径不在字典树中，提前终止
    4. 使用集合避免重复结果
    
    优化：
    - 字典树剪枝：避免无效搜索
    - 提前终止：当前路径不在字典树中时停止
    
    时间复杂度：O(mn * max_length)，其中 max_length 是最长单词长度
    空间复杂度：O(total_chars)（字典树空间）
    """
    if not board or not board[0] or not words:
        return []
    m, n = len(board), len(board[0])
    result = set()
    trie = Trie()
    for word in words:
        trie.insert(word)
    def dfs(i: int, j: int, node: TrieNode, path: str) -> None:
        char = board[i][j]
        if char not in node.children:
            return
        curr_node = node.children[char]
        curr_path = path + char
        if curr_node.is_end:
            result.add(curr_path)
        board[i][j] = '#'
        for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < m and 0 <= nj < n:
                dfs(ni, nj, curr_node, curr_path)
        board[i][j] = char
    for i in range(m):
        for j in range(n):
            dfs(i, j, trie.root, "")
    return list(result)

def wordBreak(s, wordDict):
    """单词拆分"""
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

def wordBreakII(s, wordDict):
    """单词拆分 II"""
    word_set = set(wordDict)
    n = len(s)
    dp = [[] for _ in range(n + 1)]
    dp[0] = [""]
    for i in range(1, n + 1):
        for j in range(i):
            if dp[j] and s[j:i] in word_set:
                for sentence in dp[j]:
                    dp[i].append(sentence + (" " if sentence else "") + s[j:i])
    return dp[n]