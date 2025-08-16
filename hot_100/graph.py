def numIslands(grid):
    if not grid:
        return 0
    m, n = len(grid), len(grid[0])
    count = 0
    def dfs(i, j):
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

def orangesRotting(grid):
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

def canFinish(numCourses, prerequisites):
    graph = [[] for _ in range(numCourses)]
    visited = [0] * numCourses
    for course, prereq in prerequisites:
        graph[course].append(prereq)
    def hasCycle(course):
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
    def __init__(self):
        self.children = {}
        self.is_end = False

class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True
    
    def search(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end
    
    def startsWith(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True

class WordDictionary:
    def __init__(self):
        self.root = TrieNode()
    
    def addWord(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True
    
    def search(self, word):
        def dfs(node, index):
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

def exist(board, word):
    if not board or not board[0]:
        return False
    m, n = len(board), len(board[0])
    def dfs(i, j, index):
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

def findWords(board, words):
    if not board or not board[0] or not words:
        return []
    m, n = len(board), len(board[0])
    result = set()
    trie = Trie()
    for word in words:
        trie.insert(word)
    def dfs(i, j, node, path):
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