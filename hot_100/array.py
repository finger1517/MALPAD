from typing import List

def maxSubArray(nums: List[int]) -> int:
    """
    最大子数组和 - Kadane算法
    
    问题：找到一个具有最大和的连续子数组（子数组最少包含一个元素）
    
    思路：
    1. 遍历数组，维护两个变量：current_sum 和 max_sum
    2. current_sum 表示以当前位置为结尾的子数组的最大和
    3. max_sum 表示整个数组的最大子数组和
    4. 对于每个元素，我们有两种选择：
       - 将当前元素加入到前面的子数组中（current_sum + nums[i]）
       - 从当前元素重新开始子数组（nums[i]）
    5. 选择这两种情况中较大的一个作为 current_sum
    6. 更新 max_sum
    
    时间复杂度：O(n)，空间复杂度：O(1)
    """
    max_sum = nums[0]
    current_sum = nums[0]
    for i in range(1, len(nums)):
        current_sum = max(nums[i], current_sum + nums[i])
        max_sum = max(max_sum, current_sum)
    return max_sum

def merge(intervals: List[List[int]]) -> List[List[int]]:
    """
    合并区间
    
    问题：将所有重叠的区间合并为一个区间
    
    思路：
    1. 首先按照区间的起始位置进行排序
    2. 遍历排序后的区间，对于每个区间：
       - 如果当前区间与上一个区间不重叠（当前区间的起始位置 > 上一个区间的结束位置），
         则将当前区间加入结果列表
       - 如果重叠，则合并两个区间（更新上一个区间的结束位置为两个区间结束位置的较大值）
    
    时间复杂度：O(n log n)（排序），空间复杂度：O(n)（存储结果）
    """
    intervals.sort(key=lambda x: x[0])
    merged = []
    for interval in intervals:
        if not merged or merged[-1][1] < interval[0]:
            merged.append(interval)
        else:
            merged[-1][1] = max(merged[-1][1], interval[1])
    return merged

def rotate(nums: List[int], k: int) -> None:
    """
    旋转数组
    
    问题：将数组中的元素向右移动 k 个位置
    
    思路：
    1. 使用三次反转的方法，空间复杂度为 O(1)
    2. 第一次反转整个数组
    3. 第二次反转前 k 个元素
    4. 第三次反转剩余的元素
    5. 注意：k 可能大于数组长度，所以需要先取模
    
    示例：nums = [1,2,3,4,5,6,7], k = 3
    1. 反转整个数组：[7,6,5,4,3,2,1]
    2. 反转前 3 个元素：[5,6,7,4,3,2,1]
    3. 反转后 4 个元素：[5,6,7,1,2,3,4]
    
    时间复杂度：O(n)，空间复杂度：O(1)
    """
    k = k % len(nums)
    nums.reverse()
    nums[:k] = reversed(nums[:k])
    nums[k:] = reversed(nums[k:])

def productExceptSelf(nums: List[int]) -> List[int]:
    """
    除自身以外数组的乘积
    
    问题：计算数组中除当前元素外所有元素的乘积
    
    思路：
    1. 使用两个数组：left 和 right
    2. left[i] 表示 nums[i] 左边所有元素的乘积
    3. right[i] 表示 nums[i] 右边所有元素的乘积
    4. 最终结果：result[i] = left[i] * right[i]
    5. 优化：可以只用一个结果数组，先计算左边的乘积，再从右往左计算右边的乘积
    
    时间复杂度：O(n)，空间复杂度：O(n)（可以优化到 O(1)，不算结果数组）
    """
    n = len(nums)
    left = [1] * n
    right = [1] * n
    res = [1] * n
    for i in range(1, n):
        left[i] = left[i - 1] * nums[i - 1]
    for i in range(n - 2, -1, -1):
        right[i] = right[i + 1] * nums[i + 1]
    for i in range(n):
        res[i] = left[i] * right[i]
    return res

def firstMissingPositive(nums: List[int]) -> int:
    """
    交换法
    缺失的第一个正数
    
    问题：找到数组中缺失的最小正整数（要求时间复杂度 O(n)，空间复杂度 O(1)）
    
    思路：
    1. 利用数组本身作为哈希表，将正数放到正确的位置上
    2. 遍历数组，对于每个正数 nums[i]，如果它在 1 到 n 的范围内，
       并且它不等于 nums[nums[i] - 1]，就交换它们的位置
    3. 再次遍历数组，找到第一个 nums[i] != i + 1 的位置
    4. 如果所有位置都正确，则缺失的是 n + 1
    
    示例：nums = [3,4,-1,1]
    1. 将 3 放到位置 2：[-1,4,3,1]
    2. 将 4 放到位置 3：[-1,1,3,4]
    3. 将 1 放到位置 0：[1,-1,3,4]
    4. 第一个不匹配的位置是 1，所以缺失的是 2
    
    时间复杂度：O(n)，空间复杂度：O(1)
    """
    n = len(nums)
    for i in range(n):
        while 1 <= nums[i] <= n and nums[nums[i] - 1] != nums[i]:
            nums[nums[i] - 1], nums[i] = nums[i], nums[nums[i] - 1]
    for i in range(n):
        if nums[i] != i + 1:
            return i + 1
    return n + 1