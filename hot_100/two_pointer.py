def moveZeroes(nums):
    left = 0
    for right in range(len(nums)):
        if nums[right] != 0:
            nums[left], nums[right] = nums[right], nums[left]
            left += 1

def maxArea(height):
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

def threeSum(nums):
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

def trap(height):
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