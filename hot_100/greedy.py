def maxProfit(prices):
    if not prices:
        return 0
    min_price = prices[0]
    max_profit = 0
    for price in prices:
        min_price = min(min_price, price)
        max_profit = max(max_profit, price - min_price)
    return max_profit

def canJump(nums):
    max_reach = 0
    for i in range(len(nums)):
        if i > max_reach:
            return False
        max_reach = max(max_reach, i + nums[i])
    return True

def jump(nums):
    jumps = 0
    current_end = 0
    farthest = 0
    for i in range(len(nums) - 1):
        farthest = max(farthest, i + nums[i])
        if i == current_end:
            jumps += 1
            current_end = farthest
    return jumps

def partitionLabels(s):
    last_pos = {}
    for i, char in enumerate(s):
        last_pos[char] = i
    start = 0
    end = 0
    result = []
    for i, char in enumerate(s):
        end = max(end, last_pos[char])
        if i == end:
            result.append(end - start + 1)
            start = i + 1
    return result