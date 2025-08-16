class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def inorderTraversal(root):
    res = []
    def inorder(node):
        if node:
            inorder(node.left)
            res.append(node.val)
            inorder(node.right)
    inorder(root)
    return res

def maxDepth(root):
    if not root:
        return 0
    return 1 + max(maxDepth(root.left), maxDepth(root.right))

def invertTree(root):
    if not root:
        return None
    root.left, root.right = invertTree(root.right), invertTree(root.left)
    return root

def isSymmetric(root):
    def isMirror(left, right):
        if not left and not right:
            return True
        if not left or not right:
            return False
        return (left.val == right.val and 
                isMirror(left.left, right.right) and 
                isMirror(left.right, right.left))
    return isMirror(root.left, root.right) if root else True

def diameterOfBinaryTree(root):
    diameter = 0
    def dfs(node):
        nonlocal diameter
        if not node:
            return 0
        left = dfs(node.left)
        right = dfs(node.right)
        diameter = max(diameter, left + right)
        return 1 + max(left, right)
    dfs(root)
    return diameter

def levelOrder(root):
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
    def validate(node, low=float('-inf'), high=float('inf')):
        if not node:
            return True
        if node.val <= low or node.val >= high:
            return False
        return (validate(node.left, low, node.val) and 
                validate(node.right, node.val, high))
    return validate(root)

def kthSmallest(root, k):
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
    if not preorder or not inorder:
        return None
    root_val = preorder[0]
    root = TreeNode(root_val)
    root_pos = inorder.index(root_val)
    root.left = buildTree(preorder[1:root_pos + 1], inorder[:root_pos])
    root.right = buildTree(preorder[root_pos + 1:], inorder[root_pos + 1:])
    return root

def hasPathSum(root, targetSum):
    if not root:
        return False
    if not root.left and not root.right:
        return targetSum == root.val
    return (hasPathSum(root.left, targetSum - root.val) or 
            hasPathSum(root.right, targetSum - root.val))

def lowestCommonAncestor(root, p, q):
    if not root or root == p or root == q:
        return root
    left = lowestCommonAncestor(root.left, p, q)
    right = lowestCommonAncestor(root.right, p, q)
    if left and right:
        return root
    return left if left else right

def maxPathSum(root):
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