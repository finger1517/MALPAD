from typing import List

def searchInsert(nums: List[int], target: int) -> int:
    """
    搜索插入位置
    
    问题：在排序数组中找到目标值的插入位置
    
    思路：
    1. 使用二分查找找到目标值的位置
    2. 如果找到目标值，返回其索引
    3. 如果没找到，返回左指针的位置（即应该插入的位置）
    4. 关键点：循环结束时，left 就是要插入的位置
    
    示例：nums = [1,3,5,6], target = 5
    - left=0, right=3, mid=1, nums[1]=3 < 5 -> left=2
    - left=2, right=3, mid=2, nums[2]=5 == 5 -> return 2
    
    时间复杂度：O(log n)，空间复杂度：O(1)
    """
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = (left + right) // 2
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return left

def searchMatrix(matrix: List[List[int]], target: int) -> bool:
    """
    搜索二维矩阵
    
    问题：在 m x n 的矩阵中查找目标值，矩阵具有以下特性：
    - 每行的元素从左到右升序排列
    - 每列的元素从上到下升序排列
    
    思路：
    1. 将二维矩阵展开成一维数组进行二分查找
    2. 使用 mid // n 计算行，mid % n 计算列
    3. 这样可以利用二分查找的高效性
    
    示例：matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,60]], target = 3
    - 将矩阵看作一维数组：[1,3,5,7,10,11,16,20,23,30,34,60]
    - 使用二分查找找到目标值
    
    时间复杂度：O(log(mn))，空间复杂度：O(1)
    """
    if not matrix or not matrix[0]:
        return False
    m, n = len(matrix), len(matrix[0])
    left, right = 0, m * n - 1
    while left <= right:
        mid = (left + right) // 2
        row, col = mid // n, mid % n
        if matrix[row][col] == target:
            return True
        elif matrix[row][col] < target:
            left = mid + 1
        else:
            right = mid - 1
    return False

def searchRange(nums: List[int], target: int) -> List[int]:
    """
    在排序数组中查找元素的第一个和最后一个位置
    
    问题：在排序数组中找到目标值的起始位置和结束位置
    
    思路：
    1. 使用两次二分查找：
       - 第一次查找目标值的左边界（第一个出现的位置）
       - 第二次查找目标值的右边界（最后一个出现的位置）
    2. 查找左边界时，当 nums[mid] >= target 时，右指针左移
    3. 查找右边界时，当 nums[mid] <= target 时，左指针右移
    
    示例：nums = [5,7,7,8,8,10], target = 8
    - 左边界查找：找到第一个 8 的位置（索引 3）
    - 右边界查找：找到最后一个 8 的位置（索引 4）
    
    时间复杂度：O(log n)，空间复杂度：O(1)
    """
    def find_left():
        left, right = 0, len(nums) - 1
        res = -1
        while left <= right:
            mid = (left + right) // 2
            if nums[mid] >= target:
                right = mid - 1
            else:
                left = mid + 1
            if nums[mid] == target:
                res = mid
        return res
    def find_right():
        left, right = 0, len(nums) - 1
        res = -1
        while left <= right:
            mid = (left + right) // 2
            if nums[mid] <= target:
                left = mid + 1
            else:
                right = mid - 1
            if nums[mid] == target:
                res = mid
        return res
    return [find_left(), find_right()]

def search(nums: List[int], target: int) -> int:
    """
    搜索旋转排序数组
    
    问题：在旋转排序数组中查找目标值
    
    思路：
    1. 旋转排序数组的特点：数组被分成两个有序的部分
    2. 每次二分时，判断哪一部分是有序的
    3. 如果左半部分有序：
       - 如果目标在左半部分范围内，搜索左半部分
       - 否则搜索右半部分
    4. 如果右半部分有序：
       - 如果目标在右半部分范围内，搜索右半部分
       - 否则搜索左半部分
    
    示例：nums = [4,5,6,7,0,1,2], target = 0
    - 第一次：left=0, right=6, mid=3, nums[3]=7 > target
    - 左半部分 [4,5,6,7] 有序，但 target 不在其中
    - 搜索右半部分 [0,1,2]
    
    时间复杂度：O(log n)，空间复杂度：O(1)
    """
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = (left + right) // 2
        if nums[mid] == target:
            return mid
        if nums[left] <= nums[mid]:
            if nums[left] <= target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        else:
            if nums[mid] < target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1
    return -1

def findMin(nums: List[int]) -> int:
    """
    寻找旋转排序数组中的最小值
    
    问题：在旋转排序数组中找到最小值
    
    思路：
    1. 旋转排序数组的最小值就是旋转点
    2. 使用二分查找，比较 mid 和 right 的值
    3. 如果 nums[mid] > nums[right]，说明最小值在右半部分
    4. 如果 nums[mid] <= nums[right]，说明最小值在左半部分
    5. 注意：left < right 而不是 left <= right，避免死循环
    
    示例：nums = [3,4,5,1,2]
    - left=0, right=4, mid=2, nums[2]=5 > nums[4]=2 -> left=3
    - left=3, right=4, mid=3, nums[3]=1 <= nums[4]=2 -> right=3
    - left == right，返回 nums[3] = 1
    
    时间复杂度：O(log n)，空间复杂度：O(1)
    """
    left, right = 0, len(nums) - 1
    while left < right:
        mid = (left + right) // 2
        if nums[mid] > nums[right]:
            left = mid + 1
        else:
            right = mid
    return nums[left]

def findMedianSortedArrays(nums1: List[int], nums2: List[int]) -> float:
    """
    寻找两个正序数组的中位数
    
    问题：找到两个排序数组的中位数，要求时间复杂度 O(log(m+n))
    
    思路：
    1. 使用二分查找找到分割点，使得两个数组的左半部分元素数量等于右半部分
    2. 定义 partition1 和 partition2，满足 partition1 + partition2 = (m + n + 1) // 2
    3. 需要满足的条件：nums1[partition1-1] <= nums2[partition2] 且 nums2[partition2-1] <= nums1[partition1]
    4. 如果总元素个数为奇数，中位数是 max(max_left1, max_left2)
    5. 如果总元素个数为偶数，中位数是 (max(max_left1, max_left2) + min(min_right1, min_right2)) / 2
    
    示例：nums1 = [1,3], nums2 = [2]
    - partition1 = 1, partition2 = 1
    - max_left1 = 1, min_right1 = 3
    - max_left2 = 2, min_right2 = inf
    - 满足条件，总元素数为奇数，返回 max(1, 2) = 2
    
    时间复杂度：O(log(min(m,n)))，空间复杂度：O(1)
    """
    if len(nums1) > len(nums2):
        nums1, nums2 = nums2, nums1
    m, n = len(nums1), len(nums2)
    left, right = 0, m
    while left <= right:
        partition1 = (left + right) // 2
        partition2 = (m + n + 1) // 2 - partition1
        max_left1 = float('-inf') if partition1 == 0 else nums1[partition1 - 1]
        min_right1 = float('inf') if partition1 == m else nums1[partition1]
        max_left2 = float('-inf') if partition2 == 0 else nums2[partition2 - 1]
        min_right2 = float('inf') if partition2 == n else nums2[partition2]
        if max_left1 <= min_right2 and max_left2 <= min_right1:
            if (m + n) % 2 == 0:
                return (max(max_left1, max_left2) + min(min_right1, min_right2)) / 2
            else:
                return max(max_left1, max_left2)
        elif max_left1 > min_right2:
            right = partition1 - 1
        else:
            left = partition1 + 1
    return 0