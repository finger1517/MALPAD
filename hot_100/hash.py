def twoSum(nums, target: int) -> list[int]:
    d = {}
    for i, n in enumerate(nums):
        if target - n in d:
            return [d[target - n], i]
        d[n] = i

def groupAnagrams(strs: list[str]) -> list[list[str]]:
    res = {}
    for s in strs:
        key = tuple(sorted(s))
        res.setdefault(key, []).append(s)
    return list(res.values())


def longestConsecutive(nums: list[int]) -> int:
    num_set = set(nums)
    max_len = 0
    for n in num_set:
        if n - 1 not in num_set:
            cur = n
            cnt = 1
            while cur + 1 in num_set:
                cur += 1
                cnt += 1
            max_len = max(max_len, cnt)
    return max_len

