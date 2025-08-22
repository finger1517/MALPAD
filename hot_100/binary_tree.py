from typing import List, Optional

class TreeNode:
    """二叉树节点定义"""
    def __init__(self, val: int = 0, left: Optional['TreeNode'] = None, right: Optional['TreeNode'] = None):
        self.val = val
        self.left = left
        self.right = right

def inorderTraversal(root: Optional[TreeNode]) -> List[int]:
    """
    二叉树的中序遍历
    
    问题：按照左-根-右的顺序遍历二叉树
    
    思路：
    1. 递归解法：先遍历左子树，然后访问根节点，最后遍历右子树
    2. 也可以使用迭代解法，用栈模拟递归过程
    
    示例：
       1
        \
         2
        /
       3
    中序遍历结果：[1, 3, 2]
    
    时间复杂度：O(n)，空间复杂度：O(n)
    """
    res = []
    def inorder(node):
        if node:
            inorder(node.left)
            res.append(node.val)
            inorder(node.right)
    inorder(root)
    return res

def maxDepth(root: Optional[TreeNode]) -> int:
    """
    二叉树的最大深度
    
    问题：计算二叉树的最大深度（从根节点到最远叶子节点的最长路径上的节点数）
    
    思路：
    1. 递归解法：树的深度 = max(左子树深度, 右子树深度) + 1
    2. 空树的深度为 0
    3. 也可以使用层序遍历（BFS）计算层数
    
    示例：
       3
      / \
     9  20
        / \
       15  7
    最大深度：3
    
    时间复杂度：O(n)，空间复杂度：O(h)（h为树的高度）
    """
    if not root:
        return 0
    return 1 + max(maxDepth(root.left), maxDepth(root.right))

def invertTree(root: Optional[TreeNode]) -> Optional[TreeNode]:
    """
    翻转二叉树
    
    问题：将二叉树的每个节点的左右子树进行交换
    
    思路：
    1. 递归解法：先翻转左右子树，然后交换左右子树
    2. 也可以使用层序遍历，逐层交换每个节点的左右子树
    
    示例：
       4
      / \
     2   7
    / \ / \
   1  3 6  9
    翻转后：
       4
      / \
     7   2
    / \ / \
   9  6 3  1
    
    时间复杂度：O(n)，空间复杂度：O(h)（h为树的高度）
    """
    if not root:
        return None
    root.left, root.right = invertTree(root.right), invertTree(root.left)
    return root

def isSymmetric(root: Optional[TreeNode]) -> bool:
    """
    对称二叉树
    
    问题：判断二叉树是否是镜像对称的
    
    思路：
    1. 递归解法：判断两个子树是否是镜像的
    2. 两个子树是镜像的条件：
       - 两个子树的根节点值相同
       - 左子树的左子树与右子树的右子树是镜像的
       - 左子树的右子树与右子树的左子树是镜像的
    3. 空树是对称的
    
    示例：
       1
      / \
     2   2
    / \ / \
   3  4 4  3
    对称：True
    
       1
      / \
     2   2
      \   \
       3   3
    对称：False
    
    时间复杂度：O(n)，空间复杂度：O(h)（h为树的高度）
    """
    def isMirror(left: Optional[TreeNode], right: Optional[TreeNode]) -> bool:
        if not left and not right:
            return True
        if not left or not right:
            return False
        return (left.val == right.val and 
                isMirror(left.left, right.right) and 
                isMirror(left.right, right.left))
    return isMirror(root.left, root.right) if root else True

def diameterOfBinaryTree(root: Optional[TreeNode]) -> int:
    """
    二叉树的直径
    
    问题：计算二叉树的直径（任意两个节点之间最长路径的长度）
    
    思路：
    1. 直径 = max(左子树深度 + 右子树深度)
    2. 使用深度优先搜索（DFS）遍历每个节点
    3. 对于每个节点，计算其左右子树的深度，更新最大直径
    4. 返回当前节点的深度（用于父节点计算）
    
    示例：
       1
      / \
     2   3
    / \     
   4   5    
   直径：3（路径 [4,2,1,3] 或 [5,2,1,3]）
    
    时间复杂度：O(n)，空间复杂度：O(h)（h为树的高度）
    """
    diameter = 0
    def dfs(node: Optional[TreeNode]) -> int:
        nonlocal diameter
        if not node:
            return 0
        left = dfs(node.left)
        right = dfs(node.right)
        diameter = max(diameter, left + right)
        return 1 + max(left, right)
    dfs(root)
    return diameter

def levelOrder(root: Optional[TreeNode]) -> List[List[int]]:
    """
    二叉树的层序遍历
    
    问题：按层序遍历二叉树，每层作为一个子列表返回
    
    思路：
    1. 使用队列（BFS）进行层序遍历
    2. 每次处理一层的所有节点，将它们的值收集到一个列表中
    3. 将下一层的节点加入队列
    4. 重复直到队列为空
    
    示例：
       3
      / \
     9  20
        / \
       15  7
    层序遍历：[[3], [9, 20], [15, 7]]
    
    时间复杂度：O(n)，空间复杂度：O(n)
    """
    if not root:
        return []
    from collections import deque
    queue = deque([root])
    res = []
    while queue:
        level = []
        for _ in range(len(queue)):
            node = queue.popleft()
            level.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        res.append(level)
    return res

def sortedArrayToBST(nums):
    """将有序数组转换为二叉搜索树"""
    def build(left, right):
        if left > right:
            return None
        mid = (left + right) // 2
        root = TreeNode(nums[mid])
        root.left = build(left, mid - 1)
        root.right = build(mid + 1, right)
        return root
    return build(0, len(nums) - 1)

def isValidBST(root):
    """验证二叉搜索树"""
    def validate(node, low=float('-inf'), high=float('inf')):
        if not node:
            return True
        if node.val <= low or node.val >= high:
            return False
        return (validate(node.left, low, node.val) and 
                validate(node.right, node.val, high))
    return validate(root)

def kthSmallest(root, k):
    """二叉搜索树中第K小的元素"""
    stack = []
    curr = root
    while curr or stack:
        while curr:
            stack.append(curr)
            curr = curr.left
        curr = stack.pop()
        k -= 1
        if k == 0:
            return curr.val
        curr = curr.right

def rightSideView(root):
    """二叉树的右视图"""
    if not root:
        return []
    from collections import deque
    queue = deque([root])
    res = []
    while queue:
        level = []
        for _ in range(len(queue)):
            node = queue.popleft()
            level.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        res.append(level[-1])
    return res

def flatten(root):
    """二叉树展开为链表"""
    if not root:
        return None
    curr = root
    while curr:
        if curr.left:
            prev = curr.left
            while prev.right:
                prev = prev.right
            prev.right = curr.right
            curr.right = curr.left
            curr.left = None
        curr = curr.right

def buildTree(preorder, inorder):
    """从前序与中序遍历序列构造二叉树"""
    if not preorder or not inorder:
        return None
    root_val = preorder[0]
    root = TreeNode(root_val)
    root_pos = inorder.index(root_val)
    root.left = buildTree(preorder[1:root_pos + 1], inorder[:root_pos])
    root.right = buildTree(preorder[root_pos + 1:], inorder[root_pos + 1:])
    return root

def hasPathSum(root, targetSum):
    """路径总和"""
    if not root:
        return False
    if not root.left and not root.right:
        return targetSum == root.val
    return (hasPathSum(root.left, targetSum - root.val) or 
            hasPathSum(root.right, targetSum - root.val))

def lowestCommonAncestor(root, p, q):
    """二叉树的最近公共祖先"""
    if not root or root == p or root == q:
        return root
    left = lowestCommonAncestor(root.left, p, q)
    right = lowestCommonAncestor(root.right, p, q)
    if left and right:
        return root
    return left if left else right

def maxPathSum(root):
    """二叉树中的最大路径和"""
    max_sum = float('-inf')
    def dfs(node):
        nonlocal max_sum
        if not node:
            return 0
        left = max(dfs(node.left), 0)
        right = max(dfs(node.right), 0)
        current = node.val + left + right
        max_sum = max(max_sum, current)
        return node.val + max(left, right)
    dfs(root)
    return max_sum