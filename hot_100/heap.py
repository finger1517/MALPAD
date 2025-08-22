import heapq
from typing import List

def findKthLargest(nums: List[int], k: int) -> int:
    """
    数组中的第K个最大元素
    
    问题：找出数组中第 k 个最大的元素
    
    思路：
    1. 使用堆来高效地找到第 k 大的元素
    2. 可以使用最小堆或最大堆
    3. heapq 模块提供了 nlargest 函数，直接返回前 k 大的元素
    
    堆的优势：
    - 时间复杂度 O(n log k)，比排序的 O(n log n) 更好
    - 适用于大数据集和流式数据
    
    示例：nums = [3,2,1,5,6,4], k = 2
    - nlargest(2, nums) = [6,5]
    - 返回 [6,5][-1] = 5
    
    时间复杂度：O(n log k)，空间复杂度：O(k)
    """
    return heapq.nlargest(k, nums)[-1]

def topKFrequent(nums: List[int], k: int) -> List[int]:
    """
    前K个高频元素
    
    问题：找出数组中出现频率前 k 高的元素
    
    思路：
    1. 首先统计每个数字的频率
    2. 使用堆来维护前 k 个高频元素
    3. 使用最大堆（通过存储负频率）
    4. 弹出堆顶元素 k 次
    
    堆的优势：
    - 不需要排序所有元素
    - 时间复杂度优于完全排序
    
    示例：nums = [1,1,1,2,2,3], k = 2
    - 频率统计：{1:3, 2:2, 3:1}
    - 堆中元素：[(-3,1), (-2,2), (-1,3)]
    - 弹出前2个：[1,2]
    
    时间复杂度：O(n log k)，空间复杂度：O(n)
    """
    freq_map = {}
    for num in nums:
        freq_map[num] = freq_map.get(num, 0) + 1
    heap = []
    for num, freq in freq_map.items():
        heapq.heappush(heap, (-freq, num))
    result = []
    for _ in range(k):
        if heap:
            result.append(heapq.heappop(heap)[1])
    return result

class MedianFinder:
    """
    数据流的中位数
    
    功能：
    - 添加数字到数据结构中
    - 查找当前所有数字的中位数
    
    思路：
    1. 使用两个堆：
       - 最大堆：存储较小的一半数字
       - 最小堆：存储较大的一半数字
    2. 维护平衡：最大堆的大小 <= 最小堆的大小 + 1
    3. 添加数字时，根据大小决定加入哪个堆
    4. 如果不平衡，调整两个堆的大小
    
    中位数计算：
    - 如果堆大小相等：(max_heap[0] + min_heap[0]) / 2
    - 如果不相等：max_heap[0]（较大堆的堆顶）
    
    时间复杂度：
    - addNum: O(log n)
    - findMedian: O(1)
    """
    def __init__(self):
        self.max_heap = []
        self.min_heap = []
    
    def addNum(self, num: int) -> None:
        if not self.max_heap or num <= -self.max_heap[0]:
            heapq.heappush(self.max_heap, -num)
        else:
            heapq.heappush(self.min_heap, num)
        if len(self.max_heap) > len(self.min_heap) + 1:
            heapq.heappush(self.min_heap, -heapq.heappop(self.max_heap))
        elif len(self.min_heap) > len(self.max_heap):
            heapq.heappush(self.max_heap, -heapq.heappop(self.min_heap))
    
    def findMedian(self) -> float:
        if len(self.max_heap) == len(self.min_heap):
            return (-self.max_heap[0] + self.min_heap[0]) / 2
        else:
            return -self.max_heap[0]