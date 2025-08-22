from typing import List

def moveZeroes(nums: List[int]) -> None:
    """
    移动零
    
    问题：将数组中的所有零移动到数组末尾，保持非零元素的相对顺序
    
    思路：
    1. 使用双指针：
       - left：指向当前非零元素应该放置的位置
       - right：遍历数组的指针
    2. 遍历数组，当遇到非零元素时，将其交换到 left 位置
    3. 这样可以保证非零元素的相对顺序不变
    
    双指针技巧：
    - 快慢指针：right 指针快速遍历，left 指针慢速标记位置
    - 原地操作：不需要额外空间
    
    示例：nums = [0,1,0,3,12]
    - left=0, right=0: nums[0]=0 -> 不交换
    - left=0, right=1: nums[1]=1 -> 交换 nums[0]和nums[1]: [1,0,0,3,12], left=1
    - left=1, right=2: nums[2]=0 -> 不交换
    - left=1, right=3: nums[3]=3 -> 交换 nums[1]和nums[3]: [1,3,0,0,12], left=2
    - left=2, right=4: nums[4]=12 -> 交换 nums[2]和nums[4]: [1,3,12,0,0], left=3
    
    时间复杂度：O(n)，空间复杂度：O(1)
    """
    left = 0
    for right in range(len(nums)):
        if nums[right] != 0:
            nums[left], nums[right] = nums[right], nums[left]
            left += 1

def maxArea(height: List[int]) -> int:
    """
    盛最多水的容器
    
    问题：在数组中找出两个线段，使得它们与x轴构成的容器能容纳最多的水
    
    思路：
    1. 使用双指针从两端开始
    2. 容器面积 = min(height[left], height[right]) × (right - left)
    3. 每次移动较短的线段，因为移动较短的线段可能会增加面积
    4. 移动较长的线段不会增加面积（高度受限于较短的线段）
    
    双指针策略：
    - 从两端向中间收缩
    - 每次选择移动可能带来更大收益的指针
    - 贪心策略：每次移动后都可能获得更大的面积
    
    示例：height = [1,8,6,2,5,4,8,3,7]
    - left=0, right=8: area=min(1,7)×8=8, max_area=8
    - 移动左指针（1较小）：left=1, right=8: area=min(8,7)×7=49, max_area=49
    - 移动右指针（7较小）：left=1, right=7: area=min(8,3)×6=18, max_area=49
    - 移动右指针（3较小）：left=1, right=6: area=min(8,8)×5=40, max_area=49
    - 移动左指针（8=8）：left=2, right=6: area=min(6,8)×4=24, max_area=49
    - 最终最大面积为49
    
    时间复杂度：O(n)，空间复杂度：O(1)
    """
    left, right = 0, len(height) - 1
    max_area = 0
    while left < right:
        area = min(height[left], height[right]) * (right - left)
        max_area = max(max_area, area)
        if height[left] < height[right]:
            left += 1
        else:
            right -= 1
    return max_area

def threeSum(nums: List[int]) -> List[List[int]]:
    """
    三数之和
    
    问题：找出所有不重复的三元组，使得三个数的和为0
    
    思路：
    1. 首先排序数组，便于使用双指针和去重
    2. 遍历数组，对于每个元素作为第一个数：
       - 使用双指针在后面的数组中寻找两个数的和为 -nums[i]
       - left = i + 1, right = len(nums) - 1
       - 根据和的大小调整指针位置
    3. 去重处理：
       - 跳过重复的第一个数
       - 找到解后跳过重复的第二个和第三个数
    
    双指针技巧：
    - 排序后，数组变得有序，可以利用双指针
    - 固定一个数，转化为两数之和问题
    - 时间复杂度从 O(n³) 降到 O(n²)
    
    示例：nums = [-1,0,1,2,-1,-4]
    - 排序后：[-4,-1,-1,0,1,2]
    - i=0: nums[i]=-4, target=4
      - left=1, right=5: nums[1]+nums[5]=-1+2=1 < 4 -> left++
      - left=2, right=5: nums[2]+nums[5]=-1+2=1 < 4 -> left++
      - left=3, right=5: nums[3]+nums[5]=0+2=2 < 4 -> left++
      - left=4, right=5: nums[4]+nums[5]=1+2=3 < 4 -> left++
    - i=1: nums[i]=-1, target=1
      - left=2, right=5: nums[2]+nums[5]=-1+2=1 == 1 -> 找到 [-1,-1,2]
      - left=3, right=4: nums[3]+nums[4]=0+1=1 == 1 -> 找到 [-1,0,1]
    - 结果：[[-1,-1,2], [-1,0,1]]
    
    时间复杂度：O(n²)，空间复杂度：O(1)（不算排序）
    """
    nums.sort()
    res = []
    for i in range(len(nums) - 2):
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        left, right = i + 1, len(nums) - 1
        while left < right:
            total = nums[i] + nums[left] + nums[right]
            if total < 0:
                left += 1
            elif total > 0:
                right -= 1
            else:
                res.append([nums[i], nums[left], nums[right]])
                while left < right and nums[left] == nums[left + 1]:
                    left += 1
                while left < right and nums[right] == nums[right - 1]:
                    right -= 1
                left += 1
                right -= 1
    return res

def trap(height: List[int]) -> int:
    """
    接雨水
    
    问题：计算柱状图能接多少雨水
    
    思路：
    1. 使用双指针从两端向中间遍历
    2. 维护左右两边的最大高度
    3. 对于每个位置，能接的雨水量取决于左右两边最大高度的较小值
    4. 每次移动较矮的一边，因为较矮的一边限制了雨水量
    
    双指针策略：
    - left_max：左边的最大高度
    - right_max：右边的最大高度
    - 每次比较左右两边的高度，移动较矮的一边
    - 当前位置的雨水量 = min(left_max, right_max) - height[i]
    
    示例：height = [0,1,0,2,1,0,1,3,2,1,2,1]
    - left=0, right=11: left_max=0, right_max=0
    - height[0]=0 < height[11]=1 -> 移动左边
    - left=1, left_max=1: water += 1-1=0
    - left=2, left_max=1: water += 1-0=1
    - left=3, left_max=2: water += 2-2=0
    - left=4, left_max=2: water += 2-1=1
    - 继续计算，总雨水量为6
    
    时间复杂度：O(n)，空间复杂度：O(1)
    """
    left, right = 0, len(height) - 1
    left_max, right_max = 0, 0
    water = 0
    while left < right:
        if height[left] < height[right]:
            if height[left] >= left_max:
                left_max = height[left]
            else:
                water += left_max - height[left]
            left += 1
        else:
            if height[right] >= right_max:
                right_max = height[right]
            else:
                water += right_max - height[right]
            right -= 1
    return water

def lengthOfLongestSubstring(s):
    """无重复字符的最长子串"""
    char_set = set()
    left = 0
    max_len = 0
    for right in range(len(s)):
        while s[right] in char_set:
            char_set.remove(s[left])
            left += 1
        char_set.add(s[right])
        max_len = max(max_len, right - left + 1)
    return max_len

def findAnagrams(s, p):
    """找到字符串中所有字母异位词"""
    if len(p) > len(s):
        return []
    p_count = [0] * 26
    s_count = [0] * 26
    for i in range(len(p)):
        p_count[ord(p[i]) - ord('a')] += 1
        s_count[ord(s[i]) - ord('a')] += 1
    res = []
    if p_count == s_count:
        res.append(0)
    for i in range(len(p), len(s)):
        s_count[ord(s[i]) - ord('a')] += 1
        s_count[ord(s[i - len(p)]) - ord('a')] -= 1
        if p_count == s_count:
            res.append(i - len(p) + 1)
    return res

def subarraySum(nums, k):
    """和为K的子数组"""
    count = 0
    prefix_sum = {0: 1}
    current_sum = 0
    for num in nums:
        current_sum += num
        if current_sum - k in prefix_sum:
            count += prefix_sum[current_sum - k]
        prefix_sum[current_sum] = prefix_sum.get(current_sum, 0) + 1
    return count

def minWindow(s, t):
    """最小覆盖子串"""
    if not s or not t or len(s) < len(t):
        return ""
    need = {}
    window = {}
    for char in t:
        need[char] = need.get(char, 0) + 1
    left, right = 0, 0
    valid = 0
    start = 0
    min_len = float('inf')
    while right < len(s):
        char = s[right]
        right += 1
        if char in need:
            window[char] = window.get(char, 0) + 1
            if window[char] == need[char]:
                valid += 1
        while valid == len(need):
            if right - left < min_len:
                start = left
                min_len = right - left
            char = s[left]
            left += 1
            if char in need:
                if window[char] == need[char]:
                    valid -= 1
                window[char] -= 1
    return "" if min_len == float('inf') else s[start:start + min_len]

def maxSlidingWindow(nums, k):
    """滑动窗口最大值"""
    from collections import deque
    dq = deque()
    res = []
    for i in range(len(nums)):
        while dq and nums[i] >= nums[dq[-1]]:
            dq.pop()
        dq.append(i)
        if dq[0] == i - k:
            dq.popleft()
        if i >= k - 1:
            res.append(nums[dq[0]])
    return res

def topKFrequent(nums, k):
    """前K个高频元素"""
    freq_map = {}
    for num in nums:
        freq_map[num] = freq_map.get(num, 0) + 1
    bucket = [[] for _ in range(len(nums) + 1)]
    for num, freq in freq_map.items():
        bucket[freq].append(num)
    res = []
    for i in range(len(bucket) - 1, -1, -1):
        if bucket[i]:
            res.extend(bucket[i])
            if len(res) >= k:
                break
    return res[:k]