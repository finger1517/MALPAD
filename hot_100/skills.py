from typing import List

def singleNumber(nums: List[int]) -> int:
    """
    只出现一次的数字
    
    问题：找出数组中只出现一次的数字，其他数字都出现两次
    
    思路：
    1. 使用异或运算的性质：
       - a ^ a = 0
       - a ^ 0 = a
       - 异或运算满足交换律和结合律
    2. 将所有数字进行异或运算，最终结果就是只出现一次的数字
    
    位运算技巧：
    - 异或运算可以用来消除成对出现的数字
    - 时间复杂度 O(n)，空间复杂度 O(1)
    
    示例：nums = [4,1,2,1,2]
    - 4 ^ 1 ^ 2 ^ 1 ^ 2 = 4 ^ (1 ^ 1) ^ (2 ^ 2) = 4 ^ 0 ^ 0 = 4
    
    时间复杂度：O(n)，空间复杂度：O(1)
    """
    result = 0
    for num in nums:
        result ^= num
    return result

def majorityElement(nums: List[int]) -> int:
    """
    多数元素
    
    问题：找出数组中出现次数超过 n/2 的元素
    
    思路：
    1. 使用摩尔投票算法（Boyer-Moore Voting Algorithm）
    2. 维护一个候选元素和计数器
    3. 遍历数组：
       - 如果计数器为 0，选择当前元素作为候选
       - 如果当前元素等于候选元素，计数器加 1
       - 否则计数器减 1
    4. 最终的候选元素就是多数元素
    
    算法原理：
    - 多数元素的出现次数超过其他所有元素的总和
    - 因此在投票过程中，多数元素会最终胜出
    
    示例：nums = [2,2,1,1,1,2,2]
    - count=0, candidate=None -> candidate=2, count=1
    - count=1, candidate=2 -> count=2
    - count=2, candidate=2 -> count=1
    - count=1, candidate=2 -> count=0
    - count=0, candidate=2 -> candidate=1, count=1
    - count=1, candidate=1 -> count=0
    - count=0, candidate=1 -> candidate=2, count=1
    - 返回 2
    
    时间复杂度：O(n)，空间复杂度：O(1)
    """
    count = 0
    candidate = None
    for num in nums:
        if count == 0:
            candidate = num
        count += 1 if num == candidate else -1
    return candidate

def sortColors(nums: List[int]) -> None:
    """
    颜色分类
    
    问题：将数组中的 0、1、2 进行排序（荷兰国旗问题）
    
    思路：
    1. 使用三指针法：
       - low：指向当前 0 的边界
       - mid：当前遍历的位置
       - high：指向当前 2 的边界
    2. 遍历规则：
       - 如果 nums[mid] == 0：交换到 low 区域，low 和 mid 都右移
       - 如果 nums[mid] == 1：mid 右移
       - 如果 nums[mid] == 2：交换到 high 区域，high 左移
    3. 最终数组会被分成三个区域：0、1、2
    
    三指针法：
    - 保证了 [0, low-1] 都是 0
    - [low, mid-1] 都是 1
    - [high+1, n-1] 都是 2
    
    示例：nums = [2,0,2,1,1,0]
    - 初始：low=0, mid=0, high=5
    - nums[0]=2 -> 交换 nums[0] 和 nums[5]: [0,0,2,1,1,2], high=4
    - nums[0]=0 -> 交换 nums[0] 和 nums[0]: [0,0,2,1,1,2], low=1, mid=1
    - nums[1]=0 -> 交换 nums[1] 和 nums[1]: [0,0,2,1,1,2], low=2, mid=2
    - nums[2]=2 -> 交换 nums[2] 和 nums[4]: [0,0,1,1,2,2], high=3
    - nums[2]=1 -> mid=3
    - nums[3]=1 -> mid=4 > high，结束
    
    时间复杂度：O(n)，空间复杂度：O(1)
    """
    low, mid, high = 0, 0, len(nums) - 1
    while mid <= high:
        if nums[mid] == 0:
            nums[low], nums[mid] = nums[mid], nums[low]
            low += 1
            mid += 1
        elif nums[mid] == 1:
            mid += 1
        else:
            nums[mid], nums[high] = nums[high], nums[mid]
            high -= 1

def nextPermutation(nums: List[int]) -> None:
    """
    下一个排列
    
    问题：找出给定数字序列的下一个更大的排列
    
    思路：
    1. 从后往前找到第一个下降的位置 i（nums[i] < nums[i+1]）
    2. 如果找到这样的 i：
       - 从后往前找到第一个大于 nums[i] 的元素 nums[j]
       - 交换 nums[i] 和 nums[j]
       - 反转 i+1 到末尾的部分
    3. 如果没找到这样的 i，说明是最大的排列，直接反转整个数组
    
    算法步骤：
    1. 寻找下降点：从后往前找到第一个 nums[i] < nums[i+1]
    2. 寻找交换点：从后往前找到第一个 nums[j] > nums[i]
    3. 交换两个元素
    4. 反转 i+1 到末尾的部分
    
    示例：nums = [1,2,3]
    - i=1 (nums[1]=2 < nums[2]=3)
    - j=2 (nums[2]=3 > nums[1]=2)
    - 交换：[1,3,2]
    - 反转 i+1 到末尾：已经是 [2]，无需反转
    
    示例：nums = [3,2,1]
    - 没有找到下降点，直接反转整个数组：[1,2,3]
    
    时间复杂度：O(n)，空间复杂度：O(1)
    """
    n = len(nums)
    i = n - 2
    while i >= 0 and nums[i] >= nums[i + 1]:
        i -= 1
    if i >= 0:
        j = n - 1
        while j > i and nums[j] <= nums[i]:
            j -= 1
        nums[i], nums[j] = nums[j], nums[i]
    left, right = i + 1, n - 1
    while left < right:
        nums[left], nums[right] = nums[right], nums[left]
        left += 1
        right -= 1

def findDuplicate(nums: List[int]) -> int:
    """
    寻找重复数
    
    问题：在包含 n+1 个整数的数组中，找出重复的数字
    条件：数字范围是 1 到 n，只有一个重复的数字
    
    思路：
    1. 使用 Floyd 的龟兔赛跑算法（检测环）
    2. 将数组视为链表，nums[i] 表示下一个节点的索引
    3. 快慢指针找到环的入口，即为重复的数字
    
    算法步骤：
    1. 初始化快慢指针都指向 nums[0]
    2. 快指针每次走两步，慢指针每次走一步，直到相遇
    3. 将慢指针重置到起点，两个指针都每次走一步，再次相遇的点就是重复数字
    
    原理：
    - 数组中的重复数字会导致链表中出现环
    - 环的入口就是重复的数字
    
    示例：nums = [1,3,4,2,2]
    - 链表：1 -> 3 -> 2 -> 4 -> 2 -> 4 -> ...
    - 环：2 -> 4 -> 2
    - 环的入口是 2
    
    时间复杂度：O(n)，空间复杂度：O(1)
    """
    slow = fast = nums[0]
    while True:
        slow = nums[slow]
        fast = nums[nums[fast]]
        if slow == fast:
            break
    slow = nums[0]
    while slow != fast:
        slow = nums[slow]
        fast = nums[fast]
    return slow