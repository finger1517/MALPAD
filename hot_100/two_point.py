def moveZeroes(nums: list[int]) -> None:
    """
    移动零 - 将所有0移动到数组末尾，保持非零元素相对顺序
    使用双指针，时间复杂度O(n)，空间复杂度O(1)
    """
    left = 0
    for right in range(len(nums)):
        if nums[right] != 0:
            nums[left], nums[right] = nums[right], nums[left]
            left += 1

def maxArea(height: list[int]) -> int:
    """
    盛最多水的容器 - 找出两条线，使得它们与x轴共同构成的容器可以容纳最多的水
    使用双指针，时间复杂度O(n)，空间复杂度O(1)
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

def threeSum(nums: list[int]) -> list[list[int]]:
    """
    三数之和 - 找出所有不重复的三元组，使得a + b + c = 0
    使用排序+双指针，时间复杂度O(n^2)，空间复杂度O(1)
    """
    nums.sort()
    res = []
    for i in range(len(nums) - 2):
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        left, right = i + 1, len(nums) - 1
        while left < right:
            s = nums[i] + nums[left] + nums[right]
            if s < 0:
                left += 1
            elif s > 0:
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

def trap(height: list[int]) -> int:
    """
    接雨水 - 计算柱子排列后能接多少雨水
    使用双指针，时间复杂度O(n)，空间复杂度O(1)
    """
    if not height:
        return 0
    left, right = 0, len(height) - 1
    left_max, right_max = height[left], height[right]
    res = 0
    while left < right:
        if left_max < right_max:
            left += 1
            left_max = max(left_max, height[left])
            res += left_max - height[left]
        else:
            right -= 1
            right_max = max(right_max, height[right])
            res += right_max - height[right]
    return res
